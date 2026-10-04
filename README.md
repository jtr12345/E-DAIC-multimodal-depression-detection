# E-DAIC Multimodal Depression Detection

This repository contains the code, aggregate results, and documentation for a participant-level multimodal depression detection study on Extended DAIC-WOZ (E-DAIC).

## Data policy

The E-DAIC raw data, audio, video, transcripts, labels, pretrained embeddings, and participant-level predictions are **not included** in this repository. They must not be uploaded or redistributed. Users must obtain access independently and follow the Extended DAIC-WOZ EULA.

The code expects a local dataset directory such as D:\DAIC-WOZ. Change paths before running it.

## Study progression

- run_baseline_1.py: official train/dev/test baseline
- run_baseline_2.py: five-fold participant-level baseline with fixed regularization
- run_baseline_3.py: nested three-fold selection of Logistic Regression C
- run_baseline_4.py: Linear SVM, Random Forest, and XGBoost comparison
- run_baseline_5.py: inner-fold threshold selection
- run_baseline_6.py: probability calibration and late-fusion weight selection
- extract_repo_style_embeddings.py: local RoBERTa, Wav2Vec2, and OpenFace feature extraction
- run_repo_style_embeddings.py: five-fold comparison on pretrained embeddings
- supplementary_analysis.py: seven modality combinations, mean/std, bootstrap confidence intervals, official split evaluation, and model comparison tests

## Main findings

On the current 275-participant dataset, traditional features favored Text + Linear SVM (Macro-F1 0.668, AUROC 0.714). The pretrained embedding comparison favored Logistic Regression. Text + Audio and Text + Audio + Video reached nearly identical five-fold AUROC values around 0.757, while video did not provide a stable additional gain. The one-time official test split gave three-modality Logistic Regression AUROC 0.768 and XGBoost AUROC 0.712.

These are research results on one dataset. They do not establish clinical diagnostic performance or external generalization.

## Reproduction

1. Obtain E-DAIC access and place the authorized files locally.
2. Install the packages in requirements.txt in a suitable Python environment.
3. Run the feature extraction script if pretrained embeddings are not already available.
4. Run the five-fold and supplementary analysis scripts.
5. Keep raw data, embeddings, and participant-level predictions outside this repository.

## Required citations

Any publication or presentation using this database must cite:

Gratch J, Artstein R, Lucas GM, Stratou G, Scherer S, Nazarian A, Wood R, Boberg J, DeVault D, Marsella S, Traum DR. The Distress Analysis Interview Corpus of Human and Computer Interviews. In: LREC. 2014:3123-3128.

Ringeval F, Schuller B, Valstar M, Cummins N, Cowie R, Tavabi L, Schmitt M, et al. AVEC 2019 Workshop and Challenge: State-of-Mind, Detecting Depression with AI, and Cross-Cultural Affect Recognition. Proceedings of the 9th International on Audio/Visual Emotion Challenge and Workshop. ACM; 2019:3-12.

The Extended DAIC-WOZ EULA remains controlling for all use of the database.
