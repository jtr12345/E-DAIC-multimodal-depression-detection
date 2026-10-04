# 仓库方法的公平对比

`run_repo_style_embeddings.py` 是给当前项目准备的对比入口。它使用仓库生成的三类参与者级特征：

- 文本：RoBERTa-base 的 `mean_pooled_embedding`（通常 768 维）
- 音频：Wav2Vec2-base 的 `mean_pooled_embedding`（通常 768 维）
- 视频：OpenFace 的 `video_features`（通常 104 维）

它们会在同一批参与者上对齐，使用与 `run_baseline_6.py` 相同的五折参与者级交叉验证，比较 Logistic Regression 和 XGBoost，并输出 Macro-F1、AUROC、Balanced Accuracy 和混淆矩阵。

## 目前为什么还不能直接运行

仓库 GitHub 代码不包含预计算 embedding；原始 notebook 需要先生成或下载三个 pickle 文件。本机当前没有 `transformers`，外网安装也被系统网络策略拦截，因此没有伪造结果。已有的 E-DAIC 原始文件和前六次实验结果不受影响。

## 获得三个 pickle 后的运行方式

```text
C:\Users\lenovo\anaconda3\envs\pytorch-one\python.exe \
  edaic_experiment\run_repo_style_embeddings.py \
  --text <text_data_utterance_with_labels.pkl> \
  --audio <audio_data_segment_with_labels.pkl> \
  --video <video_data_with_labels.pkl> \
  --out edaic_experiment\result_repo_style
```

输出目录中的 `metrics.json` 是总五折结果，`fold_metrics.csv` 是每折结果，`oof_predictions.csv` 保存每名参与者的折外概率。不要把它和仓库 notebook 的官方 train/dev/test 数值直接并列；应当把它和 `result_6` 放在同一个五折表中比较。

## 当前已知的公平性边界

仓库的原始实验使用官方 train/dev/test 划分，而且 README 没有给出一个完整、统一的最终数字表。因此在得到 pickle 以前，只能确认方法和代码框架，不能严谨地说仓库效果一定高于当前结果。
