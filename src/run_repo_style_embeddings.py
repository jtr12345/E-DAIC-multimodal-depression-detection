"""用仓库同款的预训练 embedding 做公平五折比较。

输入是 JonathanChan9001 仓库生成的三个 pickle：
text_data_utterance_with_labels.pkl、audio_data_segment_with_labels.pkl、
video_data_with_labels.pkl。脚本只在参与者级别对齐一次，然后使用和
run_baseline_6.py 相同的 StratifiedKFold、Macro-F1 与 AUROC 评估，避免把
仓库的官方 train/dev/test 结果和本项目的五折结果直接混在一起。

仓库的字段名是 mean_pooled_embedding、video_features、Participant_ID、
Depression_label；如果你的 pickle 字段不同，脚本会在启动时明确提示。
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit("缺少 xgboost，请先在当前 Python 环境安装 xgboost。") from exc

SEED = 42


def load_pickle(path):
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, pd.DataFrame):
        raise TypeError(f"{path} 不是 pandas DataFrame，而是 {type(obj).__name__}。")
    return obj


def vector_column(frame, column, ids):
    if column not in frame:
        raise KeyError(f"找不到字段 {column!r}。实际字段：{list(frame.columns)}")
    rows = frame.set_index("Participant_ID").loc[ids, column]
    values = [np.asarray(value, dtype=float).ravel() for value in rows]
    widths = {len(value) for value in values}
    if len(widths) != 1:
        raise ValueError(f"{column} 的 embedding 长度不一致：{sorted(widths)}")
    return np.vstack(values)


def read_inputs(text_path, audio_path, video_path, label_root):
    text = load_pickle(text_path)
    audio = load_pickle(audio_path)
    video = load_pickle(video_path)
    required = {"Participant_ID", "Depression_label"}
    for name, frame in (("text", text), ("audio", audio), ("video", video)):
        missing = required - set(frame.columns)
        if missing:
            raise KeyError(f"{name} pickle 缺少字段：{sorted(missing)}")
    ids = sorted(set(text.Participant_ID) & set(audio.Participant_ID) & set(video.Participant_ID))
    if not ids:
        raise ValueError("三个 pickle 没有共同的 Participant_ID。")
    labels = text.set_index("Participant_ID").loc[ids, "Depression_label"].to_numpy(dtype=int)
    for name, frame in (("audio", audio), ("video", video)):
        other = frame.set_index("Participant_ID").loc[ids, "Depression_label"].to_numpy(dtype=int)
        if not np.array_equal(labels, other):
            raise ValueError(f"{name} 的 Depression_label 与 text 不一致。")
    features = {
        "text": vector_column(text, "mean_pooled_embedding", ids),
        "audio": vector_column(audio, "mean_pooled_embedding", ids),
        "video": vector_column(video, "video_features", ids),
    }
    # 仓库 pickle 没有保证所有参与者都包含官方 split；若存在则保留下来便于审计。
    official = text.set_index("Participant_ID").reindex(ids).get("split")
    metadata = pd.DataFrame({"Participant_ID": ids, "label": labels})
    if official is not None:
        metadata["official_split"] = official.to_numpy()
    return features, labels, metadata


def models():
    lr = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
    )
    xgb = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        XGBClassifier(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=4, tree_method="hist",
        ),
    )
    return {"logistic_regression": lr, "xgboost": xgb}


def evaluate(y, probability):
    prediction = (probability >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "auroc": float(roc_auc_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
        "n": int(len(y)),
    }


def run(text_path, audio_path, video_path, label_root, output):
    features, y, metadata = read_inputs(text_path, audio_path, video_path, label_root)
    combined = {"text": features["text"], "audio": features["audio"], "video": features["video"]}
    combined["early_tav"] = np.hstack([features["text"], features["audio"], features["video"]])
    names = list(combined)
    results = {}
    rows = []
    oof = metadata.copy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for model_name in ("logistic_regression", "xgboost"):
        for feature_name in names:
            probability = np.full(len(y), np.nan)
            fold_rows = []
            for fold, (train, validation) in enumerate(cv.split(combined[feature_name], y), start=1):
                model = models()[model_name]
                model.fit(combined[feature_name][train], y[train])
                probability[validation] = model.predict_proba(combined[feature_name][validation])[:, 1]
                fold_rows.append({"fold": fold, "model": model_name, "features": feature_name,
                                  **evaluate(y[validation], probability[validation])})
            key = f"{model_name}__{feature_name}"
            results[key] = evaluate(y, probability)
            oof[key + "_prob"] = probability
            oof[key + "_pred"] = (probability >= 0.5).astype(int)
            rows.extend(fold_rows)
            print(key, results[key], flush=True)
    output.mkdir(parents=True, exist_ok=True)
    oof.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(output / "fold_metrics.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "本目录由 run_repo_style_embeddings.py 生成。\n\n"
        "比较的是仓库的 RoBERTa/Wav2Vec2/OpenFace embedding 在相同五折参与者级划分下的结果。\n"
        "仓库原始官方 split 结果不能与这里的五折结果直接混报。\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, default=Path(r"D:\DAIC-WOZ"),
                        help="保留参数以记录数据来源；标签从 pickle 的 Depression_label 读取。")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "result_repo_style")
    args = parser.parse_args()
    run(args.text, args.audio, args.video, args.label_root, args.out)
