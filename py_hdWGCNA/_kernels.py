"""
Optimized compute kernels for py-hdWGCNA.

Strategy:
- Use numpy BLAS for matrix multiply (adj @ adj) - already optimal
- Use Numba for element-wise operations that are slow in pure Python
- Falls back gracefully to pure numpy if numba is not available
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _rank_rows(x: np.ndarray) -> np.ndarray:
    """Rank each row of a 2D array (for Spearman correlation)."""
    from scipy.stats import rankdata

    return rankdata(x, axis=1).astype(np.float64)


def compute_tom_numba(
    adj_matrix: np.ndarray,
    tom_type: str = "signed",
    tom_denom: str = "min",
    parallel: bool = True,
) -> np.ndarray:
    """Compute TOM using optimized kernel.

    Uses numpy BLAS for the critical matrix multiply (adj @ adj),
    which is already multi-threaded and SIMD-optimized.

    Parameters
    ----------
    adj_matrix : np.ndarray
        Adjacency matrix
    tom_type : str
        'signed' or 'unsigned'
    tom_denom : str
        'min' or 'max'
    parallel : bool
        Unused (BLAS handles threading internally)

    Returns
    -------
    np.ndarray
        TOM dissimilarity matrix
    """
    adj_work = adj_matrix.astype(np.float64, copy=True)
    np.fill_diagonal(adj_work, 0.0)

    k_i = adj_work.sum(axis=1)

    if tom_denom == "min":
        denominator = np.minimum(k_i[:, np.newaxis], k_i[np.newaxis, :])
    else:
        denominator = np.maximum(k_i[:, np.newaxis], k_i[np.newaxis, :])

    denominator += 1.0
    denominator -= adj_work
    denominator[denominator < 1e-6] = 1e-6

    # Critical path: matrix multiply (numpy BLAS, already optimal)
    num = adj_work @ adj_work + adj_work
    np.fill_diagonal(num, 0.0)

    TOM = num / denominator
    np.fill_diagonal(TOM, 1.0)
    np.clip(TOM, 0.0, 1.0, out=TOM)

    dissTOM = 1.0 - TOM
    np.fill_diagonal(dissTOM, 0.0)
    np.clip(dissTOM, 0.0, 1.0, out=dissTOM)

    return dissTOM


def compute_correlation_numba(
    expr_mat: np.ndarray,
    method: str = "pearson",
    parallel: bool = True,
) -> np.ndarray:
    """Compute correlation matrix using optimized kernel.

    For Pearson: uses numpy BLAS (already optimal).
    For Spearman: uses vectorized scipy.stats.rankdata.

    Parameters
    ----------
    expr_mat : np.ndarray
        Genes x Samples matrix
    method : str
        'pearson', 'spearman', or 'bicor'
    parallel : bool
        Unused (BLAS handles threading internally)

    Returns
    -------
    np.ndarray
        Correlation matrix
    """
    x = np.asarray(expr_mat, dtype=np.float64)

    if method == "spearman":
        x = _rank_rows(x)

    if method == "bicor":
        from .utils import _bicor_vectorized

        return _bicor_vectorized(x)

    # Handle NaN
    nan_mask = np.isnan(x)
    if nan_mask.any():
        x = x.copy()
        col_means = np.nanmean(x, axis=1, keepdims=True)
        for i in range(x.shape[0]):
            x[i, nan_mask[i]] = col_means[i, 0]

    # Pearson: use numpy BLAS (already optimal)
    n_genes, n_samples = x.shape
    means = np.mean(x, axis=1, keepdims=True)
    stds = np.std(x, axis=1, ddof=1, keepdims=True)
    stds = np.where(stds < 1e-15, 1.0, stds)
    centered = x - means
    normalized = centered / stds
    cor_matrix = (normalized @ normalized.T) / (n_samples - 1)
    return np.clip(cor_matrix, -1.0, 1.0)


def is_numba_available() -> bool:
    """Check if Numba JIT is available."""
    return HAS_NUMBA
