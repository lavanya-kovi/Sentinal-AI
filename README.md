# SentinelAI: Privacy-Preserving Credit Risk Scoring API

**Course:** Secure and Private AI — Georgia State University, Spring 2026  
**Team:** Lavanya Kovi, Sirichandana Bikkasani, Sai Yaswanth Reddy Suram  
**Primary Attack-Defense Pair:** Membership Inference Attack (MIA) vs. DP-SGD

---

## Overview

SentinelAI is a credit risk scoring system that predicts loan defaults while defending against:
- **Membership Inference Attacks** — using DP-SGD (Opacus) during training
- **Model Extraction** — using API hardening (rate limiting + label-only output) at inference
- **Fairness Degradation** — audited via Fairlearn (all models pass ECOA AIR >= 0.80)

Trained on 307,511 real loan applications from the Home Credit Default Risk dataset.

---

## Project Structure

```
SentinelAI_Code/
├── SentinelAI.ipynb            # Main notebook — all experiments with outputs
├── train_baseline.py           # Standalone PyTorch training script
├── api/
│   ├── __init__.py             # Imports the FastAPI app
│   └── main.py                 # FastAPI with 5 defense layers:
│                               #   1. JWT RS256 authentication
│                               #   2. Rate limiting (50 req/hr)
│                               #   3. Label-only output (no probabilities)
│                               #   4. DP-SGD model inference
│                               #   5. PII auto-redaction in logs
├── models/
│   ├── __init__.py             # Imports CreditRiskMLP
│   └── mlp.py                  # MLP: 165→256(GroupNorm)→128→64→2
│                               # GroupNorm instead of BatchNorm for Opacus
├── results/
│   ├── phase2_results.json     # All experimental metrics
│   └── README.md               # Explains generated output files
├── requirements.txt            # All Python dependencies
└── README.md                   # This file
```

---

## Setup and Reproduction

### Option 1: Google Colab (Recommended)

**Step 1:** Upload `SentinelAI.ipynb` to [Google Colab](https://colab.research.google.com)

**Step 2:** Set GPU runtime: `Runtime → Change runtime type → T4 GPU`

**Step 3:** Run Cell 1 to install numpy:
```python
!pip install numpy==1.26.4
```
Then go to `Runtime → Restart runtime`

**Step 4:** Run all remaining cells (Cell 2 through Cell 22) in order.

**Total runtime:** ~35 minutes on T4 GPU, ~60+ minutes on CPU

### Option 2: Local Machine

```bash
# Clone or unzip this package
cd SentinelAI_Code

# Install dependencies (numpy first due to version conflicts)
pip install numpy==1.26.4
pip install -r requirements.txt

# Open notebook
jupyter notebook SentinelAI.ipynb
```

### Dataset

**Home Credit Default Risk** — 307,511 loan applications, 122 features, 8.1% default rate

Download options:
- **Kaggle:** https://www.kaggle.com/competitions/home-credit-default-risk/data
- **Google Drive (our copy):** https://drive.google.com/drive/folders/1jXcOE23QSGZf-pq9JA7qITsZMLEcxU-0

Set `DATA_DIR` in Cell 3 to the folder containing `application_train.csv`.

---

## Running the API

```bash
# Start the FastAPI service
uvicorn api.main:app --reload --port 8000

# 1. Get a JWT token
curl -X POST http://localhost:8000/token \
     -d "username=demo&password=secret"

# 2. Make a prediction (replace <token>)
curl -X POST http://localhost:8000/v1/predict \
     -H "Authorization: Bearer <token>" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.5, 0.3, 0.8, ...], "applicant_id": "APP_001"}'

# 3. Health check
curl http://localhost:8000/health
```

The API returns **label-only** output (DEFAULT_RISK or NO_DEFAULT) — no probabilities exposed.

---

## Notebook Cell Guide

| Cell | Stage | What It Does |
|------|-------|-------------|
| 1 | Setup | Install numpy==1.26.4 → **restart runtime** |
| 2 | Setup | Install all libraries (torch, opacus, fairlearn, etc.) |
| 3 | Setup | Mount Google Drive, set DATA_DIR |
| 4 | Stage 1 | Load Home Credit dataset (307,511 records) |
| 5 | Stage 1 | Preprocess: encode, impute, scale, SMOTE, split |
| 6 | Stage 2 | Define CreditRiskMLP (165→256→128→64→2) |
| 7 | Stage 2 | Train baseline MLP (no DP) |
| 8 | Stage 2 | Evaluate — **Normal Service** (AUC 0.7430) |
| 9 | Stage 2 | Train 5 DP-SGD models (ε = 0.5 to 8.0) |
| 10 | Stage 5 | Plot privacy-utility curve |
| 11 | Stage 3 | MIA attack — **Attack + Defense states** |
| 12 | Stage 5 | Plot MIA advantage bar chart |
| 13 | Stage 3 | Model extraction attack (KnockoffNets) |
| 14 | Stage 5 | Plot extraction fidelity curve |
| 15 | Stage 3 | Extraction defense interpretation |
| 16 | Stage 3 | Fairness audit (Fairlearn — AIR, EqOdds) |
| 17 | Stage 5 | Three-way trade-off plot |
| 18 | Stage 5 | Save all results to JSON |
| 19 | Stage 4 | Live FastAPI demo (real server, 5 defense layers) |
| 20 | — | Honest reflection + 7-pillar security matrix |
| 21 | Stage 5 | Updated MIA chart |
| 22 | Stage 5 | Full interactive dashboard (downloads HTML) |

---

## Key Results

### Three Service States
| State | AUC-ROC | MIA Advantage | Privacy |
|-------|---------|---------------|---------|
| Normal Service (No DP) | 0.7430 | N/A | None |
| Under Attack (MIA on baseline) | 0.7430 | 0.0049 | None |
| With Defense (DP-SGD ε=2.0) | 0.7383 | 0.0110 | ε=2.0, δ=10⁻⁵ |

### DP-SGD Privacy-Utility
| Epsilon | AUC | Drop | Literature Target |
|---------|-----|------|-------------------|
| ∞ (No DP) | 0.7430 | — | 0.76 |
| 0.5 | 0.7361 | -0.9% | 0.62 |
| 2.0 | 0.7383 | -0.6% | 0.69 |
| 8.0 | 0.7378 | -0.7% | 0.73 |

### Fairness (Gender)
| Model | AIR | EqOdds Diff | ECOA (≥0.80) |
|-------|-----|-------------|--------------|
| No DP | 1.264 | 0.148 | PASS |
| ε=0.5 | 1.043 | 0.078 | PASS |
| ε=2.0 | 1.057 | 0.105 | PASS |

---

## References

[1] Song, Shokri & Mittal, "Systematic evaluation of privacy risks," USENIX Security 2021  
[2] Choquette-Choo et al., "Label-only membership inference attacks," ICML 2021  
[3] Tramèr et al., "Stealing ML models via prediction APIs," USENIX Security 2016  
[4] Liu et al., "ML-Doctor," USENIX Security 2022  
[5] Bagdasaryan et al., "Differential privacy has disparate impact," NeurIPS 2019  
[6] Naresh & Ayyappa, "DP-SGD in federated credit scoring," Scientific Reports 2026  
[7] Abadi et al., "Deep learning with differential privacy," ACM CCS 2016
