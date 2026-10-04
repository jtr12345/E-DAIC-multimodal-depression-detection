"""补充实验：七种模态消融、五折均值/标准差、bootstrap CI、McNemar 和官方 split。

输入使用 result_repo_style 生成的三个 pickle，不读取原始数据目录中的标签来训练。
脚本输出完整 CSV/JSON，供报告引用。原始 E-DAIC 文件不写入输出目录。
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

SEED = 42
MODALITIES = ("text", "audio", "video")
COMBINATIONS = {
    "T": ("text",), "A": ("audio",), "V": ("video",),
    "TA": ("text", "audio"), "TV": ("text", "video"),
    "AV": ("audio", "video"), "TAV": MODALITIES,
}


def load(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def read_data(text_path, audio_path, video_path):
    frames = {"text": load(text_path), "audio": load(audio_path), "video": load(video_path)}
    ids = sorted(set(frames["text"].Participant_ID) & set(frames["audio"].Participant_ID) & set(frames["video"].Participant_ID))
    y = frames["text"].set_index("Participant_ID").loc[ids, "Depression_label"].to_numpy(int)
    split = frames["text"].set_index("Participant_ID").loc[ids, "split"].to_numpy(str)
    data = {}
    for name, column in (("text", "mean_pooled_embedding"), ("audio", "mean_pooled_embedding"), ("video", "video_features")):
        data[name] = np.vstack([np.asarray(v, dtype=float).ravel() for v in frames[name].set_index("Participant_ID").loc[ids, column]])
    return data, y, split, np.asarray(ids)


def make_model(name):
    if name == "logistic_regression":
        estimator = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)
    elif name == "xgboost":
        estimator = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=3, objective="binary:logistic",
                                  eval_metric="logloss", random_state=SEED, n_jobs=4, tree_method="hist")
    else:
        raise ValueError(name)
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), estimator)


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "auroc": float(roc_auc_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred))}


def bootstrap_ci(y, p, n=2000):
    rng = np.random.default_rng(SEED)
    values = {"macro_f1": [], "auroc": [], "balanced_accuracy": []}
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        m = metrics(y[idx], p[idx])
        for key in values:
            values[key].append(m[key])
    return {key: {"estimate": float(metrics(y, p)[key]),
                  "lower_95": float(np.percentile(value, 2.5)),
                  "upper_95": float(np.percentile(value, 97.5))}
            for key, value in values.items()}


def run_cv(data, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows, oof = [], {}
    for model_name in ("logistic_regression", "xgboost"):
        for combo_name, combo in COMBINATIONS.items():
            X = np.hstack([data[name] for name in combo])
            p = np.full(len(y), np.nan)
            for fold, (train, valid) in enumerate(cv.split(X, y), start=1):
                model = make_model(model_name)
                model.fit(X[train], y[train])
                p[valid] = model.predict_proba(X[valid])[:, 1]
                fold_rows.append({"model": model_name, "combination": combo_name, "fold": fold, **metrics(y[valid], p[valid])})
            oof[f"{model_name}__{combo_name}"] = p
    return pd.DataFrame(fold_rows), oof


def official_split(data, y, split):
    rows = []
    train, dev, test = split == "train", split == "dev", split == "test"
    for model_name in ("logistic_regression", "xgboost"):
        for combo_name in ("T", "TA", "TAV"):
            combo = COMBINATIONS[combo_name]
            X = np.hstack([data[name] for name in combo])
            model = make_model(model_name)
            model.fit(X[train], y[train])
            for name, mask in (("dev", dev), ("test", test)):
                p = model.predict_proba(X[mask])[:, 1]
                rows.append({"model": model_name, "combination": combo_name, "split": name,
                             "n": int(mask.sum()), **metrics(y[mask], p)})
    return pd.DataFrame(rows)


def compare_models(y, oof):
    out = []
    pairs = [("logistic_regression__TAV", "logistic_regression__T"),
             ("logistic_regression__TAV", "xgboost__TAV")]
    rng = np.random.default_rng(SEED)
    for left, right in pairs:
        pl, pr = oof[left], oof[right]
        dl = (pl >= 0.5).astype(int)
        dr = (pr >= 0.5).astype(int)
        left_only = int(((dl == 1) & (dr == 0) & (dl != y)).sum())
        right_only = int(((dl == 0) & (dr == 1) & (dr != y)).sum())
        exact = binomtest(left_only, left_only + right_only, 0.5).pvalue if left_only + right_only else 1.0
        diffs = {"macro_f1": [], "auroc": []}
        for _ in range(2000):
            idx = rng.integers(0, len(y), len(y))
            if len(np.unique(y[idx])) < 2:
                continue
            diffs["macro_f1"].append(metrics(y[idx], pl[idx])["macro_f1"] - metrics(y[idx], pr[idx])["macro_f1"])
            diffs["auroc"].append(metrics(y[idx], pl[idx])["auroc"] - metrics(y[idx], pr[idx])["auroc"])
        out.append({"left": left, "right": right, "mcnemar_left_only_errors": left_only,
                    "mcnemar_right_only_errors": right_only, "mcnemar_exact_p": float(exact),
                    "bootstrap_f1_diff_lower_95": float(np.percentile(diffs["macro_f1"], 2.5)),
                    "bootstrap_f1_diff_upper_95": float(np.percentile(diffs["macro_f1"], 97.5)),
                    "bootstrap_auroc_diff_lower_95": float(np.percentile(diffs["auroc"], 2.5)),
                    "bootstrap_auroc_diff_upper_95": float(np.percentile(diffs["auroc"], 97.5))})
    return pd.DataFrame(out)


def main(args):
    data, y, split, ids = read_data(args.text, args.audio, args.video)
    folds, oof = run_cv(data, y)
    summary = folds.groupby(["model", "combination"])[["macro_f1", "auroc", "balanced_accuracy"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    ci_rows = []
    for name, p in oof.items():
        ci = bootstrap_ci(y, p)
        ci_rows.append({"model_combination": name, **{f"{m}_{k}": v for m, values in ci.items() for k, v in values.items()}})
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output / "ablation_fold_metrics.csv", index=False)
    summary.to_csv(output / "ablation_mean_std.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(output / "bootstrap_ci.csv", index=False)
    official_split(data, y, split).to_csv(output / "official_split_final.csv", index=False)
    compare_models(y, oof).to_csv(output / "model_comparison_tests.csv", index=False)
    pd.DataFrame({"Participant_ID": ids, "label": y, "official_split": split, **{k + "_prob": v for k, v in oof.items()}}).to_csv(output / "ablation_oof_predictions.csv", index=False)
    (output / "analysis_config.json").write_text(json.dumps({"seed": SEED, "n": len(y), "n_bootstrap": 2000,
        "combinations": COMBINATIONS, "models": ["logistic_regression", "xgboost"]}, indent=2), encoding="utf-8")
    print("完成：七种模态组合 × 两种分类器，五折均值/标准差、bootstrap CI、官方 split 和模型差异检验。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "supplementary_results")
    main(parser.parse_args())
