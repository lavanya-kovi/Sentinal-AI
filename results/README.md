# Results Directory

## Pre-computed Results
- phase2_results.json — All experimental metrics (AUC, MIA, extraction, fairness)

## Generated After Running SentinelAI.ipynb
Running all cells in the notebook generates:

### Model Weights
- mlp_baseline.pt — Baseline MLP (no differential privacy)
- mlp_eps_0.5.pt — DP-SGD trained, epsilon = 0.5
- mlp_eps_1.0.pt — DP-SGD trained, epsilon = 1.0
- mlp_eps_2.0.pt — DP-SGD trained, epsilon = 2.0 (recommended)
- mlp_eps_4.0.pt — DP-SGD trained, epsilon = 4.0
- mlp_eps_8.0.pt — DP-SGD trained, epsilon = 8.0

### Plots
- privacy_utility_curve.png — AUC-ROC vs epsilon trade-off
- mia_attack_advantage.png — MIA advantage bar chart (all models)
- extraction_fidelity.png — Extraction fidelity vs query budget
- three_way_tradeoff.png — Privacy x Utility x Fairness

### Preprocessing Artifacts
- imputer.pkl — Fitted median imputer (train set only)
- scaler.pkl — Fitted MinMax scaler
- feature_names.npy — Feature column names after encoding (165 features)

### Dashboard
- SentinelAI_Dashboard.html — Interactive dashboard (opens in any browser)

## How to Reproduce
1. Open SentinelAI.ipynb in Google Colab
2. Set runtime to T4 GPU
3. Run Cell 1, restart runtime, then run all remaining cells
4. All files above will be saved to /content/results/
5. Results are also backed up to Google Drive at /MyDrive/SentinelAI_Results/
