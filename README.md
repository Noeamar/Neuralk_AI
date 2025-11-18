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


	•	prior/ – utilities from TabICL (contains GaussianNoise, XSampler, etc.)
	•	TempMLP_SCM.py – temporal SCM generator (MLP + AR + periodicity)
	•	metrics_uni.py – evaluation of one dataset (ACF, ADF, CCF, spectrum)
	•	metrics.py – evaluation of multiple datasets + diversity metrics
	•	SCM_temp_MLP.ipynb – main notebook: generation + visualisation
	•	requirements.txt – dependencies
	•	README.md – project explanation

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

- Autoregressive dependence: h_t = h_new + α * h_{t−1}
- Periodicity: h_t += β * h_{t−period}
- Gaussian noise: + ε
- Block-wise dropout init → increases structural diversity

from TempMLP_SCM import TemporalMLPSCM


## 5. Temporal Evaluation — *Analyse d’un SEUL dataset*

**But :** vérifier qu’un dataset généré contient bien un **signal temporel exploitable** (et pas du simple bruit).

📍 Localisation : `metrics_uni.py`

Ce module analyse **un dataset unique** à travers :

- **Stationnarité (ADF test)** → détecter si la série est non-stationnaire (réaliste).
- **Autocorrélation (ACF)** → vérifier la présence de dépendances temporelles.
- **Spectre de puissance (periodogram)** → détecter saisonnalité / périodicité dominante.

➡️ **Objectif final :** s’assurer que les séries générées ne sont pas du bruit pur, mais qu’elles portent un vrai *signal temporel* utilisable par un modèle d’apprentissage.

---

## 6. Dataset-Level Diversity — *Comparer PLUSIEURS datasets*

**But :** évaluer si le générateur produit **de la diversité statistique réelle** entre différents jeux de données – indispensable pour constituer un corpus de *pre-training* pour un foundation model.

📍 Localisation : `metrics.py`

Chaque dataset est résumé en une **signature statistique** composée de trois volets :

| Aspect analysé  | Ce qui est mesuré |
|-----------------|------------------|
| **Marginal**    | moyenne, variance, skewness, kurtosis, quantiles… |
| **Temporel**    | valeurs d’ACF à différents lags, decoherence time, fréquence dominante… |
| **Structure**   | corrélations entre variables (flattened correlation matrix) |

À partir de ces signatures :
- une **matrice de distances pairwise** est calculée entre datasets ;
- puis des **indicateurs de diversité** sont extraits :
  - `mean_pairwise_distance` → diversité moyenne
  - `min_pairwise_distance` → datasets similaires
  - `max_pairwise_distance` → datasets très différents
  - `corr_mean`, `corr_std` → diversité structurelle (corrélations)

**Objectif final :** déterminer si le pipeline est capable de générer des **scénarios variés, cohérents et réalistes** – une propriété essentielle pour le pré-entraînement d’un *time series foundation model*.