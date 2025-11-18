# 🧠 TempTabFM – Synthetic Time Series Tabular Data Generator  
_A temporal extension of the TabICL SCM designed for pre-training time series tabular foundation models._

---

## 📌 1. Project Overview

This repository implements a **Synthetic Time Series Tabular Data Generator**, inspired by the **Structural Causal Model (SCM)** from **TabICL**, but extended with:

✓ **Temporal dependencies** (autoregressive memory → `α`)  
✓ **Periodicity / seasonality** (`β`, lagged memory)  
✓ **Gaussian noise** for dataset diversity  
✓ **Hyperparameter sensitivity study**  
✓ **Evaluation framework** for dataset-level quality & diversity  

This work is designed for a **TempTabFM** research context:  
> _“How do we generate enough high-quality temporal data to pre-train a foundation model for tabular time series?”_

---

## 📁 2. Repository Structure 

.
├── prior/                      # Copied from TabICL (required)
│   ├── utils/                 # GaussianNoise, XSampler
│   └── …
├── TempMLP_SCM.py             # Core temporal SCM generator
├── metrics_uni.py             # Evaluate ONE dataset (temporal signal)
├── metrics.py                 # Compare several datasets (diversity)
├── SCM_temp_MLP.ipynb         # Full benchmark + visualisation
├── requirements.txt
└── README.md

---

## 3. Installation

bash
python -m venv .venv_temp_scm
source .venv_temp_scm/bin/activate        # Linux/Mac
# or
.\.venv_temp_scm\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt

---

## 4. Core Class — TemporalMLPSCM

Located in: TempMLP_SCM.py

✔ Autoregressive dependence: h_t = h_new + α * h_{t−1}
✔ Periodicity: h_t += β * h_{t−period}
✔ Gaussian noise: + ε
✔ Block-wise dropout init → increases structural diversity

from TempMLP_SCM import TemporalMLPSCM

model = TemporalMLPSCM(
    seq_len=100,
    num_features=10,
    num_causes=10,
    num_layers=4,
    hidden_dim=32,
    alpha=0.3,
    beta=1.2,
    period=20,
    use_periodicity=True,
    device="cpu",
)

X, y = model.forward()
print(X.shape)   # (100, 10)

X, y = model.generate_dataset(n_individuals=50)
print(X.shape)    # (50 * seq_len , num_features)

## 5. Temporal Evaluation (ONE dataset)

Located in metrics_uni.py

from metrics_uni import evaluate_dataset_temporality
evaluate_dataset_tempority(X)

## 6. Dataset-Level Diversity (MULTIPLE datasets)

Located in: metrics.py

from metrics import (
    dataset_signature, compute_correlation_signature,
    pairwise_distances, diversity_metrics,
    plot_all_diversity
)

X_list = []
for _ in range(10):
    X, y = model.generate_dataset(50)    # ⚠ mêmes hyperparams
    X_list.append(X)

# Extract signatures
sigs = [dataset_signature(X)["combined"] for X in X_list]
corr_sigs = [compute_correlation_signature(X) for X in X_list]

# Distance matrices
D_global = pairwise_distances(sigs)
D_corr = compute_pairwise_corr_distances(corr_sigs)

# Diversity indicators
print(diversity_metrics(D_global))
print(summarize_corr_diversity(D_corr))

# Plots
plot_all_diversity(D_global, D_corr)