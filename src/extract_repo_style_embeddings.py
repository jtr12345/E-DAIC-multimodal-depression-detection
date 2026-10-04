"""从本地 E-DAIC 生成可供公平对比脚本读取的三类 embedding。

文本使用 roberta-base，音频使用 facebook/wav2vec2-base；视频使用本地
OpenFace 数值列的均值和标准差。每位参与者只保存一个向量，避免把同一
参与者的多个片段同时分到不同交叉验证折中。
"""
import argparse
import pickle
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer, Wav2Vec2Model, Wav2Vec2Processor


def text_embedding(model, tokenizer, text, device):
    parts = [text[i:i + 1800] for i in range(0, max(len(text), 1), 1800)]
    vectors = []
    with torch.inference_mode():
        for part in parts:
            batch = tokenizer(part, return_tensors="pt", truncation=True, max_length=512)
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            vectors.append((hidden * mask).sum(1) / mask.sum(1).clamp_min(1))
    return torch.cat(vectors, dim=0).mean(0).cpu().numpy().astype("float32")


def audio_embedding(model, processor, audio_path, device):
    audio, rate = sf.read(audio_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != 16000:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
    chunk = 20 * 16000
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(audio), chunk):
            piece = audio[start:start + chunk]
            if len(piece) < 1600:
                continue
            inputs = processor(piece, sampling_rate=16000, return_tensors="pt", padding=True)
            values = inputs.input_values.to(device)
            hidden = model(values).last_hidden_state.mean(1)
            vectors.append(hidden.cpu())
    if not vectors:
        return np.zeros(model.config.hidden_size, dtype="float32")
    return torch.cat(vectors, dim=0).mean(0).numpy().astype("float32")


def video_features(path):
    frame = pd.read_csv(path, low_memory=False)
    if "success" in frame:
        frame = frame.loc[frame["success"] == 1]
    numeric = frame.select_dtypes(include=[np.number]).drop(
        columns=["frame", "timestamp", "frameTime", "confidence", "success"], errors="ignore"
    )
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return np.concatenate([numeric.mean().fillna(0).to_numpy(), numeric.std().fillna(0).to_numpy()]).astype("float32")


def labels(root):
    pieces = []
    for split in ("train", "dev", "test"):
        part = pd.read_csv(root / "labels" / f"{split}_split.csv")
        part["official_split"] = split
        pieces.append(part)
    frame = pd.concat(pieces, ignore_index=True)
    if frame["Participant_ID"].duplicated().any():
        raise ValueError("标签中有重复参与者。")
    return frame


def run(root, output, limit=None):
    label_frame = labels(root)
    if limit:
        label_frame = label_frame.head(limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, participants={len(label_frame)}", flush=True)
    text_model = AutoModel.from_pretrained("roberta-base").to(device).eval()
    text_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device).eval()
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    texts, audios, videos = [], [], []
    for number, row in enumerate(label_frame.itertuples(index=False), start=1):
        pid = int(row.Participant_ID)
        folder = root / "data" / f"{pid}_P"
        transcript = pd.read_csv(folder / f"{pid}_Transcript.csv")
        text = transcript["Text"].fillna("").astype(str).str.cat(sep=" ")
        texts.append({"Participant_ID": pid, "Depression_label": int(row.PHQ_Binary),
                      "split": row.official_split if hasattr(row, "official_split") else "" ,
                      "mean_pooled_embedding": text_embedding(text_model, text_tokenizer, text, device).tolist()})
        audios.append({"Participant_ID": pid, "Depression_label": int(row.PHQ_Binary),
                       "split": row.official_split if hasattr(row, "official_split") else "",
                       "mean_pooled_embedding": audio_embedding(audio_model, audio_processor, folder / f"{pid}_AUDIO.wav", device).tolist()})
        videos.append({"Participant_ID": pid, "Depression_label": int(row.PHQ_Binary),
                       "split": row.official_split if hasattr(row, "official_split") else "",
                       "video_features": video_features(folder / "features" / f"{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv").tolist()})
        print(f"{number}/{len(label_frame)} participant {pid}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("text_data_utterance_with_labels.pkl", texts), ("audio_data_segment_with_labels.pkl", audios), ("video_data_with_labels.pkl", videos)):
        with open(output / name, "wb") as handle:
            pickle.dump(pd.DataFrame(rows), handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"D:\DAIC-WOZ"))
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "repo_embeddings")
    parser.add_argument("--limit", type=int, default=None, help="调试时只处理前N人")
    args = parser.parse_args()
    run(args.root, args.out, args.limit)
