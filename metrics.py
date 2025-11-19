import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.signal import periodogram

# ============================================================
# UTILITIES
# ============================================================

def to_numpy(x):
    """Convert torch or numpy tensor to numpy array."""
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def acf_1d(x, max_lag=50):
    """Compute the autocorrelation function for a 1D signal."""
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    acf = result[result.size // 2:]
    acf = acf[:max_lag+1]
    acf /= acf[0] if acf[0] != 0 else 1
    return acf


# ============================================================
# SIGNATURE EXTRACTION
# ============================================================

def dataset_signature(X, max_lag=50):
    """
    Compute marginal, temporal, and structural signatures for dataset X (T,D).

    Returns:
        {
            "marginal": vec_marginal,
            "temporal": vec_temporal,
            "structure": vec_structure,
            "combined": concatenation of all (NON NORMALISÉ)
        }
    """
    X = to_numpy(X)
    T, D = X.shape

    marginals = []
    temporals = []

    for j in range(D):
        xj = X[:, j]

        # ----- Marginals
        m  = xj.mean()
        s  = xj.std()
        sk = skew(xj)
        ku = kurtosis(xj)
        q10, q50, q90 = np.percentile(xj, [10, 50, 90])
        marginals.extend([m, s, sk, ku, q10, q50, q90])

        # ----- Temporal
        acf = acf_1d(xj, max_lag=max_lag)
        selected_lags = [l for l in [1,5,10,20,50] if l <= max_lag]
        acf_feats = [acf[l] for l in selected_lags]

        abs_acf = np.abs(acf)
        if np.any(abs_acf < 1/np.e):
            decoh = np.argmax(abs_acf < 1/np.e)
        else:
            decoh = max_lag

        freqs, psd = periodogram(xj)
        main_freq = freqs[np.argmax(psd)]

        temporals.extend(acf_feats + [decoh, main_freq])

    # ----- Structure (Corr matrix)
    C = np.corrcoef(X, rowvar=False)
    iu = np.triu_indices(D, k=1)
    corr_vec = C[iu]

    combined = np.concatenate([marginals, temporals, corr_vec])
    return {
        "marginal": np.array(marginals),
        "temporal": np.array(temporals),
        "structure": np.array(corr_vec),
        "combined": combined
    }


# ============================================================
# NORMALISATION DES SIGNATURES
# ============================================================

def normalize_signatures(signature_list):
    """
    Normalise toutes les signatures COLONNE PAR COLONNE
    (min-max scaling) pour éviter qu'une métrique domine les autres.
    """
    sigs = np.vstack([s["combined"] for s in signature_list])  # (N, Dtotal)
    mins = sigs.min(axis=0)
    maxs = sigs.max(axis=0)
    denom = (maxs - mins) + 1e-8  # avoid div by zero

    normalized = (sigs - mins) / denom
    return normalized


# ============================================================
# DISTANCES & DIVERSITY METRICS
# ============================================================

def pairwise_distances(vectors):
    """Compute NxN matrix of L2 distances between flattened vectors."""
    vectors = [np.asarray(v).reshape(-1) for v in vectors]
    N = len(vectors)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(vectors[i] - vectors[j])
    return D


def diversity_metrics(D):
    """Compute basic statistics over the upper triangle of a distance matrix."""
    vals = D[np.triu_indices_from(D, k=1)]
    return {
        "mean_pairwise_distance": float(np.mean(vals)),
        "min_pairwise_distance": float(np.min(vals)),
        "max_pairwise_distance": float(np.max(vals)),
        "std_pairwise_distance": float(np.std(vals)),
    }


# ============================================================
# CORRELATION SIGNATURE DISTANCE
# ============================================================

def compute_correlation_signature(X):
    """Flattened upper triangular correlation matrix."""
    X = to_numpy(X)
    C = np.corrcoef(X, rowvar=False)
    tri = C[np.triu_indices_from(C, k=1)]
    return tri


def compute_pairwise_corr_distances(corr_sigs):
    N = len(corr_sigs)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i, j] = np.linalg.norm(corr_sigs[i] - corr_sigs[j])
    return D


def summarize_corr_diversity(D):
    vals = D[np.triu_indices_from(D, k=1)]
    return {
        "corr_mean": float(np.mean(vals)),
        "corr_min": float(np.min(vals)),
        "corr_max": float(np.max(vals)),
        "corr_std": float(np.std(vals)),
    }


# ============================================================
# VISUALISATIONS
# ============================================================

def plot_distance_matrix(D, title):
    plt.figure(figsize=(7,6))
    sns.heatmap(D, cmap="viridis", annot=False)
    plt.title(title)
    plt.show()


def plot_all_diversity(D_global, D_corr):
    plot_distance_matrix(D_global, "Global signature distances")
    plot_distance_matrix(D_corr, "Correlation structure distances")