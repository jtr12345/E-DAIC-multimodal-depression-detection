"""第6次实验：五折验证 + 训练集内部选择晚期融合权重。

阅读顺序：第1段 → 第8段。数据始终按参与者划分。
这是一项探索性实验；不能把反复比较同一批数据当成独立最终测试。
"""

# 第1段：导入工具。Path管理路径；numpy/pandas处理数据；sklearn训练模型。
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

SEED = 42
MODALITIES = ("text", "audio", "video")


# 第2段：读取标签。一行表示一个参与者；ID用于对应文本、音频和视频。
def read_labels(root):
    pieces = []
    for split in ("train", "dev", "test"):
        frame = pd.read_csv(root / "labels" / f"{split}_split.csv")
        frame["official_split"] = split
        pieces.append(frame)
    labels = pd.concat(pieces, ignore_index=True)
    if labels["Participant_ID"].duplicated().any():
        raise ValueError("标签中出现重复参与者，停止实验以防止身份混淆。")
    return labels


# 第3段：把每人的长序列压缩为统计量。这里不读取任何抑郁标签。
def summarize(path, separator=",", visual=False):
    frame = pd.read_csv(path, sep=separator, low_memory=False)
    if visual:
        # 丢弃人脸检测失败的帧；这是预先固定的质量规则。
        if "success" in frame:
            frame = frame.loc[frame["success"] == 1]
    frame = frame.select_dtypes(include=[np.number])
    metadata = ["frame", "timestamp", "frameTime", "confidence", "success"]
    frame = frame.drop(columns=metadata, errors="ignore")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    return pd.concat([
        frame.mean().add_suffix("_mean"),
        frame.std().add_suffix("_std"),
    ])


def read_data(root, labels):
    texts, audio_rows, video_rows = [], [], []
    for position, pid in enumerate(labels["Participant_ID"], start=1):
        folder = root / "data" / f"{pid}_P"
        transcript = pd.read_csv(folder / f"{pid}_Transcript.csv")
        if "Text" not in transcript:
            raise ValueError(f"{pid} 转录文件没有Text列。")
        # 当前E-DAIC文件没有speaker列；本版沿用提供的Text，不凭空区分说话人。
        texts.append(transcript["Text"].fillna("").astype(str).str.cat(sep=" "))
        audio_rows.append(summarize(
            folder / "features" / f"{pid}_OpenSMILE2.3.0_egemaps.csv", ";"
        ))
        video_rows.append(summarize(
            folder / "features" / f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv",
            visual=True,
        ))
        if position % 25 == 0 or position == len(labels):
            print(f"读取数据 {position}/{len(labels)}", flush=True)
    return {
        "text": np.asarray(texts),
        "audio": pd.DataFrame(audio_rows).to_numpy(dtype=float),
        "video": pd.DataFrame(video_rows).to_numpy(dtype=float),
    }


# 第4段：建立三个模型。Pipeline保证所有学习步骤只看当前训练数据。
def create_models():
    text_pipeline = make_pipeline(
        TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TruncatedSVD(n_components=50, random_state=SEED),
        LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
    )
    # SVM原始输出是分数；交叉验证校准把它转换为可参与融合的概率。
    # TF-IDF/SVD也在每个校准训练折内拟合。
    text = CalibratedClassifierCV(text_pipeline, method="sigmoid", cv=3)
    audio = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED),
    )
    video = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            max_features="sqrt", n_jobs=4, random_state=SEED,
        ),
    )
    return {"text": text, "audio": audio, "video": video}


# 第5段：仅在外层训练者内部产生预测，作为选择权重的依据。
def inner_predictions(data, target, train_indices):
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    probabilities = np.full((len(train_indices), 3), np.nan)
    for inner_train, inner_validation in inner_cv.split(train_indices, target[train_indices]):
        models = create_models()
        for column, modality in enumerate(MODALITIES):
            model = models[modality]
            model.fit(data[modality][train_indices[inner_train]], target[train_indices[inner_train]])
            probabilities[inner_validation, column] = model.predict_proba(
                data[modality][train_indices[inner_validation]]
            )[:, 1]
    assert np.isfinite(probabilities).all()
    return probabilities


def select_weights(probabilities, target):
    # 只尝试21组简单权重：每个权重为0、0.2、…、1，三者之和为1。
    # 权重允许为0：如果某个模态没带来收益，就不强行使用它。
    best_score, best_weights = -1.0, None
    for text_parts in range(6):
        for audio_parts in range(6 - text_parts):
            video_parts = 5 - text_parts - audio_parts
            weights = np.array([text_parts, audio_parts, video_parts]) / 5
            score = f1_score(target, probabilities @ weights >= 0.5, average="macro")
            if score > best_score:
                best_score, best_weights = score, weights
    return best_weights, best_score


# 第6段：计算指标。所有分类都使用预先固定的0.5阈值。
def evaluate(target, probabilities):
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "macro_f1": float(f1_score(target, predicted, average="macro")),
        "auroc": float(roc_auc_score(target, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(target, predicted)),
        "confusion_matrix": confusion_matrix(target, predicted, labels=[0, 1]).tolist(),
        "n": len(target),
    }


# 第7段：外层五折。验证者不参与特征学习、概率校准或权重选择。
def run_experiment(root, output):
    output.mkdir(parents=True, exist_ok=True)
    labels = read_labels(root)
    data = read_data(root, labels)
    target = labels["PHQ_Binary"].to_numpy(dtype=int)
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    names = [*MODALITIES, "late_equal", "late_weighted"]
    oof = {name: np.full(len(target), np.nan) for name in names}
    fold_ids = np.zeros(len(target), dtype=int)
    folds, weight_records = [], []

    for fold, (train, validation) in enumerate(outer_cv.split(target, target), start=1):
        print(f"第{fold}/5折：{len(train)}人训练，{len(validation)}人验证", flush=True)
        inner = inner_predictions(data, target, train)
        weights, inner_f1 = select_weights(inner, target[train])
        models = create_models()
        validation_probabilities = np.empty((len(validation), 3))
        for column, modality in enumerate(MODALITIES):
            model = models[modality]
            model.fit(data[modality][train], target[train])
            probabilities = model.predict_proba(data[modality][validation])[:, 1]
            oof[modality][validation] = probabilities
            validation_probabilities[:, column] = probabilities
        oof["late_equal"][validation] = validation_probabilities.mean(axis=1)
        oof["late_weighted"][validation] = validation_probabilities @ weights
        fold_ids[validation] = fold
        weight_records.append({
            "fold": fold, "text_weight": weights[0], "audio_weight": weights[1],
            "video_weight": weights[2], "inner_selection_f1": inner_f1,
        })
        for name in names:
            folds.append({"fold": fold, "model": name, **evaluate(target[validation], oof[name][validation])})
        print(f"第{fold}折完成，文本/音频/视频权重={weights.tolist()}", flush=True)

    # 每名参与者只在自己被留出的那一折得到预测。
    assert (fold_ids > 0).all() and all(np.isfinite(p).all() for p in oof.values())
    summary = {name: evaluate(target, oof[name]) for name in names}
    table = labels[["Participant_ID", "official_split"]].copy()
    table["label"] = target
    table["fold"] = fold_ids
    for name in names:
        table[name + "_prob"] = oof[name]
        table[name + "_pred"] = (oof[name] >= 0.5).astype(int)
    table.to_csv(output / "oof_predictions.csv", index=False)
    pd.DataFrame(folds).to_csv(output / "fold_metrics.csv", index=False)
    pd.DataFrame(weight_records).to_csv(output / "fusion_weights.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


# 第8段：运行入口。命令行指定数据路径，默认将结果写入脚本旁的result_6。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\DAIC-WOZ"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "result_6")
    args = parser.parse_args()
    run_experiment(args.root, args.out)
