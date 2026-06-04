# SentinelAI — Privacy-Preserving Credit-Risk Scoring API

A production-style machine-learning system that predicts loan defaults while actively defending against the attacks that threaten financial AI: **membership inference** (recovering whether a specific person's record was used in training) and **model extraction** (cloning a proprietary model by querying its API). The project goes beyond a single model — it implements a layered defense, simulates the attacks it claims to stop, and measures the cost of each defense in accuracy, privacy, and fairness on **307,511 real loan applications** from the Home Credit Default Risk dataset.

> **The core result:** combining differential privacy with API hardening protects a credit model against both privacy and intellectual-property attacks — *and* improves its fairness — at **less than 1% loss** in predictive performance.

---

## Why this problem matters

Credit-scoring models are trained on highly sensitive personal data (income, debt history, demographics) and are exposed to the public through prediction APIs. That combination creates two distinct risks that are usually studied in isolation:

- **Privacy leakage.** Models tend to *memorize* high-dimensional training records. An attacker who can query the model can often infer whether a particular individual was in the training set — a direct breach of personal privacy.
- **Intellectual-property theft.** By systematically querying the API and recording outputs, an attacker can train a *surrogate* model that mimics the original, stealing the institution's investment without ever touching the source.

SentinelAI treats these as a single, connected problem and applies a **defense-in-depth** strategy that addresses both at once, while keeping the model accurate enough to be useful and fair enough to align with lending regulation.

---

## Headline results

| Metric | Baseline (no defense) | With defenses (ε = 2.0) | Interpretation |
|---|---|---|---|
| **AUC-ROC** (predictive utility) | 0.7430 | 0.7383 | < 1% loss under a formal privacy guarantee |
| **Membership-inference advantage** | 0.0049 | ≈ 0 | Formal differential-privacy bound neutralizes leakage |
| **Adverse Impact Ratio** (1.0 = parity) | 1.264 | 1.103 (1.043 at ε=0.5) | Privacy training acts as a *fairness regularizer* |
| **Model-extraction fidelity** (lower = harder to clone) | 92.95% | 90.87% | Label-only output degrades a cloning attack |
| **Effective time to extract** (10,000 queries) | minutes | **200+ hours** | Rate limiting makes API theft impractical |

---

## Full privacy–utility–fairness sweep

The model was retrained across a range of privacy budgets (ε) to map the trade-off precisely rather than assert it:

| Metric | Baseline | ε = 8.0 | ε = 2.0 | ε = 0.5 |
|---|---|---|---|---|
| AUC-ROC | 0.7430 | 0.7378 | 0.7383 | 0.7361 |
| Test accuracy (%) | 68.29 | 68.05 | 68.10 | 67.92 |
| Membership-inference advantage | 0.0049 | ≈ 0 | ≈ 0 | ≈ 0 |
| Adverse Impact Ratio | 1.264 | 1.154 | 1.103 | 1.043 |

**Two findings stand out:**

1. **Large datasets absorb privacy noise.** With N > 300,000 records, the calibrated noise that differential privacy injects is averaged away almost entirely. The usual privacy-versus-accuracy trade-off — severe on small datasets — nearly disappears here. The cost of a strong guarantee (ε = 2.0) is under one percentage point of AUC.
2. **Privacy improved fairness.** Tightening the privacy budget steadily pushed the Adverse Impact Ratio from 1.264 toward 1.0 (parity). By smoothing out the biased outlier patterns the model would otherwise latch onto, differential privacy doubled as a fairness regularizer — moving the model into the 0.80–1.25 band associated with fair-lending compliance (ECOA).

The cost of the strongest setting (ε = 0.5) was purely in training time: it required roughly **40% more epochs** to converge because of the heavier noise on each gradient step, while final accuracy stayed competitive.

---

## How it works

### The model
A seven-layer multilayer perceptron mapping **165 engineered financial features → 256 → 128 → 64 → 2** (default / no-default). A deliberate architectural choice: **Group Normalization is used instead of Batch Normalization**, because BatchNorm mixes statistics across samples in a batch, which corrupts the per-sample gradient accounting that differential privacy depends on. GroupNorm normalizes within each sample independently, keeping the privacy guarantee valid.

### The data pipeline
- **Feature engineering** — categorical variables encoded via frequency mapping and one-hot encoding into a 165-dimensional input space.
- **Informative missingness** — for features with more than 30% missing values, binary "missingness indicator" columns are generated so the model can treat the *absence* of data as a signal rather than noise.
- **Normalization** — all numeric features scaled to [0, 1] to keep gradients stable under noise injection.
- **Class balancing** — the dataset's natural 8.1% default rate is heavily imbalanced, so SMOTE synthetically oversamples the minority class to 25%, expanding the training set to 263,840 examples and helping the model converge even with privacy noise added.

### The privacy defense (DP-SGD)
During training, each sample's gradient is **clipped** to a maximum norm, then **calibrated Gaussian noise** is added before the update. This provides a formal, mathematically provable privacy budget (ε): the smaller the ε, the stronger the guarantee that no single training record measurably influences the model — which is exactly what defeats a membership-inference attack.

### The deployment defense (defense-in-depth)
The trained model is served through a hardened API with five independent layers, each closing a different attack surface:

1. **JWT authentication** — only credentialed clients can reach the model.
2. **Rate limiting** (50 requests/hour) — converts a fast extraction attack into a multi-day one, long enough for intrusion detection to intervene.
3. **Label-only output** — returns the final decision and withholds the confidence scores that surrogate-training attacks rely on.
4. **DP-SGD model** — the trained weights already carry the formal privacy guarantee.
5. **PII redaction in logs** — sensitive request fields are scrubbed before anything is written to disk.

---

## How the defenses were validated

Defenses were not assumed to work — they were attacked:

- **Membership inference** was simulated using confidence-based attacks against both the undefended and DP-trained models; the measured attacker advantage dropped from 0.0049 to effectively zero under differential privacy.
- **Model extraction** was simulated by training surrogate models against the API under different defense configurations; label-only output reduced the surrogate's fidelity, and rate limiting extended the practical time-to-clone past 200 hours.
- **Fairness** was audited with the Adverse Impact Ratio across every privacy setting, evaluating demographic parity in lending decisions.

---

## What this project demonstrates

- **Applied ML security** — implementing *and rigorously evaluating* defenses against real attack classes, not just describing them.
- **Responsible / fair AI** — quantitative bias auditing and reasoning about regulatory alignment as a first-class concern, not an afterthought.
- **End-to-end ownership** — taking a model from raw 300K-row data through training, privacy hardening, fairness analysis, and a deployable API.
- **Sound experimental method** — sweeping a parameter (ε) to characterize a trade-off precisely instead of relying on a single data point.

---

## Tech stack

**ML & privacy:** Python, PyTorch, Opacus (DP-SGD), scikit-learn, imbalanced-learn (SMOTE)
**Security evaluation & fairness:** Adversarial Robustness Toolbox, Fairlearn, NumPy, Pandas
**API & deployment:** FastAPI, Uvicorn, JWT authentication, rate limiting
**Visualization:** Matplotlib, Seaborn

---

## Repository contents

The full implementation, including every experiment with its plots and outputs, lives in **`SentinelAI.ipynb`**, which renders directly on GitHub — the clearest way to review the actual work. The API service (`api/`), model definition (`models/`), standalone training script, and experimental metrics (`results/`) are also included.

> *Note: the Home Credit dataset is not redistributed here; it is available from Kaggle under its own terms.*

---

## Limitations and future direction

- The current feature set caps predictive power at roughly 0.74 AUC; incorporating richer signals (credit-bureau history, prior application data) is the most direct path toward a higher ceiling.
- Fairness is evaluated via the Adverse Impact Ratio; future work would add complementary metrics (demographic parity difference, calibration across subgroups) for a fuller equity picture.
- The extraction defenses raise the cost of cloning but do not make it impossible — they are designed to buy time for detection, consistent with a defense-in-depth philosophy.

---
