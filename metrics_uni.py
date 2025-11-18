import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, ccf, adfuller
from scipy.signal import periodogram


# ============================================================
# 1) AUTO-CORRÉLATION (ACF)
# ============================================================
def compute_acf(series, nlags=20):
    """
    Retourne l'ACF d'une série 1D.
    """
    series = np.asarray(series)
    return acf(series, nlags=nlags, fft=True)


def plot_acf(series, nlags=100, title="ACF"):
    acf_vals = np.abs(compute_acf(series, nlags))
    plt.stem(range(len(acf_vals)), acf_vals)
    plt.axhline(0, color='black')
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("ACF")
    plt.show()

def acf_1d(x, max_lag):
    x = x - x.mean()
    autocorr = np.correlate(x, x, mode="full")
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr[:max_lag+1]
    return autocorr / autocorr[0]

# ============================================================
# 2) CORRÉLATION CROISÉE ENTRE DEUX VARIABLES (CCF)
# ============================================================
def compute_ccf(series1, series2, nlags=20):
    """
    Retourne la CCF (corrélation croisée).
    """
    return ccf(series1, series2)[:nlags+1]


def plot_ccf(series1, series2, nlags=20, title="CCF"):
    ccf_vals = compute_ccf(series1, series2, nlags)
    plt.stem(range(len(ccf_vals)), ccf_vals)
    plt.axhline(0, color='black')
    plt.title(title)
    plt.xlabel("Lag")
    plt.ylabel("CCF")
    plt.show()



# ============================================================
# 3) TEST DE STATIONNARITÉ (ADF)
# ============================================================
def adf_test(series):
    """
    Retourne la p-value de l'ADF.
    """
    series = np.asarray(series)
    result = adfuller(series, autolag='AIC')
    return result[1]  # p-value



# ============================================================
# 4) SPECTRE DE PUISSANCE (PERIODOGRAMME)
# ============================================================
def compute_spectrum(series, fs=1.0):
    """
    Retourne (freq, power) d'un périodogramme.
    """
    freq, power = periodogram(series, fs=fs)
    return freq, power


def plot_spectrum(series, fs=1.0, title="Spectre de puissance"):
    freq, power = compute_spectrum(series, fs)
    plt.plot(freq, power)
    plt.title(title)
    plt.xlabel("Fréquence")
    plt.ylabel("Puissance")
    plt.show()



# ============================================================
# 5) PIPELINE D’ÉVALUATION (MINIMAL)
# ============================================================
def evaluate_dataset_temporality(X, max_features_to_plot=3):
    """
    X : numpy array ou torch tensor de shape (T, D)
    """

    if not isinstance(X, np.ndarray):
        X = X.detach().cpu().numpy()   # ← CORRECTION

    T, D = X.shape
    print(f"Dataset shape: {T} timesteps × {D} features\n")

    # ----------- ADF -----------
    print("=== Stationnarité (ADF) ===")
    for i in range(min(5, D)):
        pval = adf_test(X[:, i])
        print(f"Feature {i} → p-value ADF = {pval:.4f}")
    print()

    # ----------- ACF -----------
    print("=== Exemple ACF ===")
    for i in range(min(max_features_to_plot, D)):
        plot_acf(X[:, i], title=f"ACF feature {i}")

    # ----------- CCF entre feat0 et feat1 si possible -----------
    if D >= 2:
        print("\n=== Exemple CCF (feature 0 vs 1) ===")
        plot_ccf(X[:, 0], X[:, 1], title="CCF feat0 vs feat1")

    # ----------- Spectre -----------
    print("\n=== Spectre de puissance ===")
    for i in range(min(max_features_to_plot, D)):
        plot_spectrum(X[:, i], title=f"Spectre feature {i}")

    print("\nÉvaluation terminée.")

def pairwise_distances(signatures, weights=(1.0, 1.0, 1.0)):
    n = len(signatures)
    w_m, w_t, w_s = weights
    Dmat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            si, sj = signatures[i], signatures[j]
            d_m = np.linalg.norm(si["marginal"]  - sj["marginal"])
            d_t = np.linalg.norm(si["temporal"]  - sj["temporal"])
            d_s = np.linalg.norm(si["structure"] - sj["structure"])
            d   = w_m * d_m + w_t * d_t + w_s * d_s
            Dmat[i, j] = Dmat[j, i] = d
    return Dmat

def diversity_metrics(Dmat):
    # Dmat : (N,N)
    n = Dmat.shape[0]
    tril = Dmat[np.tril_indices(n, k=-1)]
    return {
        "mean_pairwise_distance": tril.mean(),
        "min_pairwise_distance":  tril.min(),
        "max_pairwise_distance":  tril.max(),
        "std_pairwise_distance":  tril.std(),
    }