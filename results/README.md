# Aggregate results

These files contain aggregate metrics only. Participant-level predictions and raw E-DAIC files are excluded.

- result_1_metrics.json through result_6_metrics.json: versioned baseline results
- result_repo_style_metrics.json: pretrained embedding results
- ablation_mean_std.csv: seven combinations and two classifiers, five-fold mean/std
- bootstrap_ci.csv: participant-level out-of-fold bootstrap 95% intervals
- official_split_final.csv: one-time official dev/test check
- model_comparison_tests.csv: bootstrap metric differences and exact McNemar tests
- analysis_config.json: seed and combination definitions
