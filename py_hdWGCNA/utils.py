"""
Utility functions for py-hdWGCNA.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy import stats


def check_wgcna_name(adata, wgcna_name):
    if wgcna_name is None:
        if 'hdWGCNA' in adata.uns:
            wgcna_name = adata.uns['hdWGCNA'].get('active_wgcna', 'hdWGCNA')
        else:
            wgcna_name = 'hdWGCNA'
    if 'hdWGCNA' not in adata.uns or wgcna_name not in adata.uns['hdWGCNA']:
        raise ValueError(f"hdWGCNA experiment '{wgcna_name}' not found in adata.uns['hdWGCNA']")
    return wgcna_name


def get_hdWGCNA_data(adata, wgcna_name=None):
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    return adata.uns['hdWGCNA'][wgcna_name]


def set_hdWGCNA_data(adata, data, wgcna_name=None):
    if 'hdWGCNA' not in adata.uns:
        adata.uns['hdWGCNA'] = {}
    if wgcna_name is None:
        wgcna_name = data.get('active_wgcna', 'hdWGCNA')
    adata.uns['hdWGCNA'][wgcna_name] = data
    adata.uns['hdWGCNA']['active_wgcna'] = wgcna_name
    return adata


def _bicor_vectorized(x: np.ndarray) -> np.ndarray:
    """
    Compute biweight midcorrelation matrix matching R WGCNA::bicor behavior.
    
    Algorithm:
    1. Compute median and MAD for each row
    2. Compute biweight weights u_i = (x_i - median) / (9 * MAD)
    3. Zero out weights where |u_i| >= 1
    4. Compute weighted covariance using biweight U-statistics
    
    Parameters
    ----------
    x : np.ndarray
        Genes x Samples matrix
    
    Returns
    -------
    np.ndarray
        Bicor correlation matrix (genes x genes)
    """
    n_genes, n_samples = x.shape
    
    medians = np.median(x, axis=1, keepdims=True)
    centered = x - medians
    mads = np.median(np.abs(centered), axis=1, keepdims=True)
    
    mads_safe = np.where(mads < 1e-15, 1.0, mads)
    
    u = centered / (9.0 * mads_safe)
    
    u_sq = u ** 2
    a_sq = np.where(u_sq < 1.0, (1.0 - u_sq) ** 2, 0.0)
    
    sum_a_sq = a_sq.sum(axis=1, keepdims=True)
    sum_a_sq_safe = np.where(sum_a_sq < 1e-15, 1.0, sum_a_sq)
    
    z = centered * a_sq / sum_a_sq_safe
    
    z_sq_sum = (z ** 2).sum(axis=1, keepdims=True)
    z_norm = np.sqrt(z_sq_sum)
    z_norm_safe = np.where(z_norm < 1e-15, 1.0, z_norm)
    
    z_normalized = z / z_norm_safe
    
    chunk_size = 2000
    if n_genes <= chunk_size:
        bicor = z_normalized @ z_normalized.T
    else:
        bicor = np.empty((n_genes, n_genes), dtype=np.float64)
        for i in range(0, n_genes, chunk_size):
            i_end = min(i + chunk_size, n_genes)
            for j in range(i, n_genes, chunk_size):
                j_end = min(j + chunk_size, n_genes)
                block = z_normalized[i:i_end] @ z_normalized[j:j_end].T
                bicor[i:i_end, j:j_end] = block
                if i != j:
                    bicor[j:j_end, i:i_end] = block.T
    
    np.clip(bicor, -1.0, 1.0, out=bicor)
    
    return bicor


def compute_correlation_matrix(expr_mat: np.ndarray, method='pearson') -> np.ndarray:
    """
    Compute correlation matrix matching R's cor() / bicor() behavior.

    Uses vectorized numpy operations for O(n^2*d) complexity.

    Parameters
    ----------
    expr_mat : np.ndarray
        Genes x Samples matrix
    method : str
        'pearson', 'spearman', or 'bicor'

    Returns
    -------
    np.ndarray
        Correlation matrix (genes x genes)
    """
    n_genes, n_samples = expr_mat.shape

    if method == 'spearman':
        from scipy.stats import rankdata
        ranked = np.zeros_like(expr_mat)
        for i in range(n_genes):
            ranked[i, :] = rankdata(expr_mat[i, :])
        expr_mat = ranked
        method = 'pearson'

    expr_mat = np.asarray(expr_mat, dtype=np.float64)
    
    nan_mask = np.isnan(expr_mat)
    if nan_mask.any():
        expr_mat = expr_mat.copy()
        col_means = np.nanmean(expr_mat, axis=1, keepdims=True)
        for i in range(n_genes):
            expr_mat[i, nan_mask[i]] = col_means[i, 0]

    if method == 'bicor':
        return _bicor_vectorized(expr_mat)

    means = np.mean(expr_mat, axis=1, keepdims=True)
    stds = np.std(expr_mat, axis=1, ddof=1, keepdims=True)
    stds_safe = np.where(stds < 1e-15, 1.0, stds)

    centered = expr_mat - means
    normalized = centered / stds_safe

    cor_matrix = (normalized @ normalized.T) / (n_samples - 1)
    cor_matrix = np.clip(cor_matrix, -1.0, 1.0)

    return cor_matrix


def soft_threshold(cor_matrix: np.ndarray, power: int, 
                   network_type: str = 'signed') -> np.ndarray:
    """
    Apply soft-thresholding to correlation matrix to create adjacency matrix.
    
    Matches R WGCNA's adjacency.fromSimilarity behavior.
    
    Parameters
    ----------
    cor_matrix : np.ndarray
        Correlation matrix
    power : int
        Soft-thresholding power
    network_type : str
        'signed', 'unsigned', or 'signed hybrid'
    
    Returns
    -------
    np.ndarray
        Adjacency matrix
    """
    if network_type == 'unsigned':
        adj = np.abs(cor_matrix) ** power
    elif network_type == 'signed':
        adj = ((1 + cor_matrix) / 2) ** power
    elif network_type == 'signed hybrid':
        pos_cor = np.where(cor_matrix > 0, cor_matrix, 0)
        adj = pos_cor ** power
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    return adj


def compute_tom(adj_matrix: np.ndarray, tom_type: str = 'signed', tom_denom: str = 'min') -> np.ndarray:
    n = adj_matrix.shape[0]

    adj_work = adj_matrix.astype(np.float64, copy=True)
    np.fill_diagonal(adj_work, 0.0)

    k_i = adj_work.sum(axis=1)

    if tom_denom == 'min':
        denominator = np.minimum(k_i[:, np.newaxis], k_i[np.newaxis, :])
    else:
        denominator = np.maximum(k_i[:, np.newaxis], k_i[np.newaxis, :])

    denominator += 1.0
    denominator -= adj_work
    denominator[denominator < 1e-6] = 1e-6

    num = adj_work @ adj_work + adj_work
    np.fill_diagonal(num, 0.0)

    TOM = num / denominator
    np.fill_diagonal(TOM, 1.0)
    np.clip(TOM, 0.0, 1.0, out=TOM)

    dissTOM = 1.0 - TOM
    np.fill_diagonal(dissTOM, 0.0)
    np.clip(dissTOM, 0.0, 1.0, out=dissTOM)

    return dissTOM


def hclust_and_cut(tom_dissim: np.ndarray, deepSplit: int = 4,
                   pamRespectsDendro: bool = False, pamStage: bool = True,
                   minClusterSize: int = 30, cutHeight: float = None,
                   method: str = 'average', verbose: int = 0) -> np.ndarray:
    n = tom_dissim.shape[0]

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform as _sq
    cond_dist = _sq(tom_dissim, checks=False)
    Z = linkage(cond_dist, method=method)
    del cond_dist

    import dynamicTreeCut as _dtc
    _func = _dtc.cutreeHybrid
    if 'df_apply' not in _func.__globals__:
        _func.__globals__['df_apply'] = _dtc.df_apply
    kwargs = dict(
        deepSplit=deepSplit,
        minClusterSize=minClusterSize,
        pamRespectsDendro=pamRespectsDendro,
        pamStage=pamStage,
        verbose=0
    )
    if cutHeight is not None:
        kwargs['cutHeight'] = cutHeight
    result_py = _func(Z, distM=tom_dissim, **kwargs)
    if isinstance(result_py, dict) and 'labels' in result_py:
        labels = np.asarray(result_py['labels']).flatten()
    elif isinstance(result_py, dict):
        for key in ['labels', 'assignment', 'clusters']:
            if key in result_py:
                labels = np.asarray(result_py[key]).flatten()
                break
        else:
            labels = np.zeros(n, dtype=int)
    else:
        labels = np.asarray(result_py).flatten()

    if len(labels) != n:
        raise ValueError(f"Label count mismatch: got {len(labels)}, expected {n}")
    return labels


def dynamic_tree_cut(merge_obj, deepSplit=4,
                     pamRespectsDendro: bool = False,
                     pamStage: bool = True,
                     minClusterSize: int = 30,
                     cutHeight: float = None,
                     verbose: int = 0):
    link = merge_obj['Z']
    n = merge_obj['n']

    import dynamicTreeCut as _dtc
    _func = _dtc.cutreeHybrid
    if 'df_apply' not in _func.__globals__:
        _func.__globals__['df_apply'] = _dtc.df_apply
    distM = merge_obj.get('distM')
    kwargs = dict(
        deepSplit=deepSplit,
        minClusterSize=minClusterSize,
        pamRespectsDendro=pamRespectsDendro,
        pamStage=pamStage,
        verbose=0
    )
    if cutHeight is not None:
        kwargs['cutHeight'] = cutHeight
    result_py = _func(link, distM=distM, **kwargs)
    labels = result_py['labels']

    if len(labels) != n:
        raise ValueError(f"Label count mismatch: got {len(labels)}, expected {n}")
    return labels


def compute_module_eigengenes(expr_mat: np.ndarray, module_labels: np.ndarray,
                               exclude_grey: bool = True) -> tuple:
    """
    Compute module eigengenes (first principal component of each module).
    
    Matches R hdWGCNA::ModuleEigengenes behavior:
    1. Z-score scale each gene (matching Seurat ScaleData)
    2. PCA on scaled data
    3. Flip eigengene sign if cor(averageExpr, PC1) < 0
       Note: averageExpr is computed on the ORIGINAL (unscaled) data,
       matching R's GetAssayData(layer='data') after ScaleData.
    
    Parameters
    ----------
    expr_mat : np.ndarray
        Genes x Samples expression matrix
    module_labels : np.ndarray
        Module assignment for each gene
    exclude_grey : bool
        Whether to exclude grey (unassigned) module
    
    Returns
    -------
    tuple
        (MEs: np.ndarray, varExplained: np.ndarray, valid_modules: np.ndarray)
        MEs is modules x samples matrix
    """
    from sklearn.decomposition import PCA

    first_appearance_order = []
    seen = set()
    for lbl in module_labels:
        if exclude_grey and lbl == 0:
            continue
        if lbl not in seen:
            first_appearance_order.append(lbl)
            seen.add(lbl)
    
    unique_modules = np.array(first_appearance_order)
    
    n_modules = len(unique_modules)
    n_samples = expr_mat.shape[1]
    
    MEs = np.zeros((n_modules, n_samples), dtype=np.float64)
    var_explained = np.zeros(n_modules, dtype=np.float64)
    
    valid_modules = []
    for i, mod in enumerate(unique_modules):
        gene_mask = module_labels == mod
        gene_idx = np.where(gene_mask)[0]
        
        if len(gene_idx) < 2:
            continue
        
        mod_expr = expr_mat[gene_idx, :]

        mod_expr_centered = mod_expr - np.nanmean(mod_expr, axis=1, keepdims=True)
        mod_expr_std = np.nanstd(mod_expr, axis=1, ddof=1, keepdims=True)
        mod_expr_std = np.where(mod_expr_std < 1e-10, 1.0, mod_expr_std)
        mod_expr_scaled = mod_expr_centered / mod_expr_std

        nan_mask = np.isnan(mod_expr_scaled)
        if nan_mask.any():
            mod_expr_scaled[nan_mask] = 0.0

        try:
            from scipy.sparse.linalg import svds
            U, s, Vt = svds(mod_expr_scaled.T, k=1)
            me = U[:, 0] * s[0]
            total_var = np.sum(mod_expr_scaled**2)
            var_ratio = np.sum(me**2) / total_var if total_var > 0 else 0.0
        except Exception:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            me = pca.fit_transform(mod_expr_scaled.T).flatten()
            var_ratio = pca.explained_variance_ratio_[0]

        avg_expr = np.nanmean(mod_expr, axis=0)
        avg_expr_centered = avg_expr - np.nanmean(avg_expr)
        me_centered = me - np.nanmean(me)
        denom = (np.sqrt(np.sum(avg_expr_centered**2) + 1e-30) *
                 np.sqrt(np.sum(me_centered**2) + 1e-30))
        pca_cor = np.sum(avg_expr_centered * me_centered) / denom
        
        if pca_cor < 0:
            me = -me
        
        MEs[i, :] = me
        var_explained[i] = var_ratio
        valid_modules.append(mod)
    
    MEs = MEs[:len(valid_modules), :]
    var_explained = var_explained[:len(valid_modules)]
    
    return MEs, var_explained, np.array(valid_modules)


def compute_kme(expr_mat: np.ndarray, MEs: np.ndarray, method: str = 'pearson') -> np.ndarray:
    n_genes = expr_mat.shape[0]
    n_modules = MEs.shape[0]
    n_samples = expr_mat.shape[1]

    expr_nan_mask = np.isnan(expr_mat)
    if expr_nan_mask.any():
        expr_mat = expr_mat.copy()
        expr_mat[expr_nan_mask] = 0.0

    MEs_nan_mask = np.isnan(MEs)
    if MEs_nan_mask.any():
        MEs = MEs.copy()
        MEs[MEs_nan_mask] = 0.0

    if method == 'bicor':
        def _bicor_z(x):
            medians = np.median(x, axis=1, keepdims=True)
            centered = x - medians
            mads = np.median(np.abs(centered), axis=1, keepdims=True)
            mads_safe = np.where(mads < 1e-15, 1.0, mads)
            u = centered / (9.0 * mads_safe)
            a_sq = np.where(np.abs(u) < 1.0, (1.0 - u**2)**2, 0.0)
            sum_a_sq = a_sq.sum(axis=1, keepdims=True)
            sum_a_sq_safe = np.where(sum_a_sq < 1e-15, 1.0, sum_a_sq)
            z = centered * a_sq / sum_a_sq_safe
            z_norm = np.sqrt((z**2).sum(axis=1, keepdims=True))
            z_norm_safe = np.where(z_norm < 1e-15, 1.0, z_norm)
            return z / z_norm_safe

        expr_z = _bicor_z(expr_mat)
        MEs_z = _bicor_z(MEs)

        kME = np.empty((n_genes, n_modules), dtype=np.float64)
        chunk_size = 2000
        for i in range(0, n_genes, chunk_size):
            i_end = min(i + chunk_size, n_genes)
            kME[i:i_end] = expr_z[i:i_end] @ MEs_z.T
    else:
        expr_centered = expr_mat - expr_mat.mean(axis=1, keepdims=True)
        expr_std = expr_centered.std(axis=1, ddof=1, keepdims=True)
        expr_std[expr_std < 1e-10] = 1.0

        MEs_centered = MEs - MEs.mean(axis=1, keepdims=True)
        MEs_std = MEs_centered.std(axis=1, ddof=1, keepdims=True)
        MEs_std[MEs_std < 1e-10] = 1.0

        MEs_z = MEs_centered / MEs_std

        kME = np.empty((n_genes, n_modules), dtype=np.float64)
        chunk_size = 2000
        for i in range(0, n_genes, chunk_size):
            i_end = min(i + chunk_size, n_genes)
            expr_chunk = expr_centered[i:i_end] / expr_std[i:i_end]
            kME[i:i_end] = (expr_chunk @ MEs_z.T) / (n_samples - 1)

    kME = np.clip(kME, -1.0, 1.0)

    return kME


def scale_free_fit_index(k: np.ndarray, power: int, nBreaks: int = 10, removeFirst: bool = False) -> float:
    result = scale_free_fit_index_full(k, nBreaks, removeFirst)
    return result['R2']


def scale_free_fit_index_full(k: np.ndarray, nBreaks: int = 10, removeFirst: bool = False) -> dict:
    k = np.asarray(k, dtype=np.float64).ravel()
    n = len(k)
    if n < 4:
        return {'R2': 0.0, 'slope': 0.0, 'truncated_R2': 0.0}

    dx = k.max() - k.min()
    if dx == 0:
        dx = abs(k.min())
    if dx == 0:
        dx = 1.0

    discretized_k = pd.cut(k, nBreaks, labels=False, include_lowest=True, right=True)

    dk = np.full(nBreaks, np.nan)
    p_dk = np.full(nBreaks, np.nan)

    grouped_mean = pd.Series(k).groupby(discretized_k).mean()
    grouped_count = pd.Series(k).groupby(discretized_k).count()

    for bin_id in grouped_mean.index:
        bid = int(bin_id)
        if 0 <= bid < nBreaks:
            dk[bid] = grouped_mean[bin_id]
            p_dk[bid] = grouped_count[bin_id] / n

    breaks1 = np.linspace(k.min(), k.max(), nBreaks + 1)
    dk2 = (breaks1[:-1] + breaks1[1:]) / 2.0

    dk = np.where(np.isnan(dk), dk2, dk)
    dk = np.where(dk == 0, dk2, dk)
    p_dk = np.where(np.isnan(p_dk), 0, p_dk)

    log_dk = np.log10(dk)

    if removeFirst and len(p_dk) > 1:
        p_dk = p_dk[1:]
        log_dk = log_dk[1:]

    log_p_dk = np.log10(p_dk + 1e-9)

    if len(log_dk) < 3:
        return {'R2': 0.0, 'slope': 0.0, 'truncated_R2': 0.0}

    try:
        X1 = np.column_stack([np.ones(len(log_dk)), log_dk])
        beta1 = np.linalg.lstsq(X1, log_p_dk, rcond=None)[0]
        p_pred1 = X1 @ beta1
        ss_res1 = np.sum((log_p_dk - p_pred1) ** 2)
        ss_tot1 = np.sum((log_p_dk - np.mean(log_p_dk)) ** 2)
        r_sq = max(0.0, 1.0 - ss_res1 / ss_tot1) if ss_tot1 > 0 else 0.0
        slope = beta1[1]
    except Exception:
        slope = 0.0
        r_sq = 0.0

    try:
        X2 = np.column_stack([np.ones(len(log_dk)), log_dk, 10.0 ** log_dk])
        beta2 = np.linalg.lstsq(X2, log_p_dk, rcond=None)[0]
        p_pred2 = X2 @ beta2
        ss_res2 = np.sum((log_p_dk - p_pred2) ** 2)
        ss_tot2 = np.sum((log_p_dk - np.mean(log_p_dk)) ** 2)
        n_params = X2.shape[1]
        n_obs = len(log_dk)
        adj_r_sq = 1.0 - (ss_res2 / ss_tot2) * (n_obs - 1) / (n_obs - n_params) if ss_tot2 > 0 and n_obs > n_params else 0.0
        truncated_r_sq = max(0.0, adj_r_sq)
    except Exception:
        truncated_r_sq = r_sq

    return {'R2': r_sq, 'slope': slope, 'truncated_R2': truncated_r_sq}


def load_r_outputs(data_dir):
    """
    Load R hdWGCNA output CSV files with proper format handling.

    Handles known R output format quirks:
    - power_table.csv: columns use dots (e.g. SFT.R.sq, not SFT.Rsq)
    - MEs.csv: NO numeric row index; last column is 'cell' (barcode)
    - kME.csv: gene_name is row index
    - modules.csv: gene_name is a regular column

    Parameters
    ----------
    data_dir : str
        Path to directory containing R output CSV files

    Returns
    -------
    dict
        Dictionary with keys: 'power_table', 'modules', 'kme', 'mes',
        each containing a properly-formatted pandas DataFrame
    """
    data_dir = os.path.abspath(data_dir)

    r_power = pd.read_csv(os.path.join(data_dir, 'power_table.csv'))

    r_modules = pd.read_csv(os.path.join(data_dir, 'modules.csv'))

    r_kme = pd.read_csv(os.path.join(data_dir, 'kME.csv'), index_col=0)

    kme_cols = [c for c in r_kme.columns if c.lower().startswith('kme') and c.lower() != 'module']
    rename_map = {}
    for c in kme_cols:
        mod_name = c
        if c.lower().startswith('kme'):
            mod_name = c[3:] if len(c) > 3 else c
            if not mod_name.startswith('_'):
                mod_name = '_' + mod_name
        new_name = 'kME' + mod_name
        if new_name != c:
            rename_map[c] = new_name
    if rename_map:
        r_kme = r_kme.rename(columns=rename_map)

    r_mes = pd.read_csv(os.path.join(data_dir, 'MEs.csv'))
    if 'cell' in r_mes.columns:
        r_mes = r_mes.set_index('cell')

    return {
        'power_table': r_power,
        'modules': r_modules,
        'kme': r_kme,
        'mes': r_mes,
    }


def load_test_data(data_dir):
    """
    Load test expression data from CSV files into an AnnData object.

    Expects files in data_dir:
    - expression_matrix.csv: genes x cells (genes in rows, cells in columns)
    - metadata.csv: cell metadata with barcode column

    Parameters
    ----------
    data_dir : str
        Path to directory containing test data CSV files

    Returns
    -------
    anndata.AnnData
        AnnData object ready for hdWGCNA pipeline
    """
    data_dir = os.path.abspath(data_dir)

    import anndata

    expr_df = pd.read_csv(os.path.join(data_dir, 'expression_matrix.csv'), index_col=0)
    meta_df = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))

    if 'cell' in meta_df.columns or 'barcode' in meta_df.columns:
        col = 'cell' if 'cell' in meta_df.columns else 'barcode'
        meta_df = meta_df.set_index(col).loc[expr_df.columns]

    adata = anndata.AnnData(X=expr_df.T.values, obs=meta_df, var=pd.DataFrame(index=expr_df.index))
    adata.var_names_make_unique()

    return adata


def benchmark_compare(adata, r_outputs, wgcna_name='hdWGCNA'):
    """
    Compare py-hdWGCNA results against R reference outputs.

    Computes correlation metrics for:
    - Power table (scale-free fit R^2)
    - Module assignments (ARI / NMI)
    - Module eigengenes hMEs (Pearson r per module)
    - kME hub gene connectivity (Pearson r per module)

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object with completed hdWGCNA results in adata.uns
    r_outputs : dict
        Output from load_r_outputs()
    wgcna_name : str
        Name of hdWGCNA experiment in adata.uns

    Returns
    -------
    dict
        Dictionary with comparison results for each metric category
    """
    from scipy.stats import pearsonr as _pearsonr
    hdW = get_hdWGCNA_data(adata, wgcna_name)
    results = {}

    r_power = r_outputs['power_table']
    py_sft = hdW.get('power_table', None)
    if py_sft is not None:
        if isinstance(py_sft, pd.DataFrame):
            py_sft_df = py_sft.copy()
            if 'SFT.R.sq' not in py_sft_df.columns:
                for c in py_sft_df.columns:
                    if 'Rsq' in c or 'R2' in c or 'SFT' in c:
                        py_sft_df = py_sft_df.rename(columns={c: 'SFT.R.sq'})
                        break
            if 'Power' in py_sft_df.columns and 'SFT.R.sq' in r_power.columns:
                merged = pd.merge(r_power, py_sft_df, on='Power', suffixes=('_R', '_Py'), how='inner')
                sq_r = [c for c in merged.columns if c.endswith('_R') and 'SFT' in c]
                sq_py = [c for c in merged.columns if c.endswith('_Py') and 'SFT' in c]
                if len(sq_r) > 0 and len(sq_py) > 0:
                    valid = merged.dropna(subset=[sq_r[0], sq_py[0]])
                    if len(valid) >= 2:
                        r_val, p_val = _pearsonr(valid[sq_r[0]], valid[sq_py[0]])
                        results['power'] = {'r': round(r_val, 4), 'p': round(p_val, 4), 'n': len(valid)}
                    else:
                        results['power'] = {'r': np.nan, 'error': f'insufficient overlap: {len(valid)}'}
                else:
                    results['power'] = {'r': np.nan, 'error': f'SFT columns not found: R_cols={[c for c in merged.columns if "SFT" in c]}'}
            else:
                results['power'] = {'r': np.nan, 'error': f'Py power_table cols: {list(py_sft_df.columns[:5])}'}
        else:
            results['power'] = {'r': np.nan, 'error': f'unexpected type: {type(py_sft).__name__}'}
    else:
        results['power'] = {'r': np.nan, 'error': 'no power_table found'}

    r_mods = r_outputs['modules']
    py_mods = hdW.get('module_colors', None)
    if py_mods is None:
        py_mods = hdW.get('module_labels', None)
    if py_mods is None:
        py_mods = hdW.get('module_table', None)
    if py_mods is not None:
        if isinstance(py_mods, pd.DataFrame):
            mods_series = py_mods.set_index(py_mods.columns[0]).iloc[:, 0] if py_mods.shape[1] >= 2 else pd.Series(py_mods.iloc[:, 0].values)
        elif isinstance(py_mods, dict):
            mods_series = pd.Series(py_mods)
        elif isinstance(py_mods, np.ndarray):
            mods_series = pd.Series(py_mods, index=adata.var_names[:len(py_mods)])
        elif isinstance(py_mods, list):
            mods_series = pd.Series(py_mods, index=adata.var_names[:len(py_mods)])
        elif isinstance(py_mods, pd.Series):
            mods_series = py_mods
        else:
            mods_series = pd.Series(py_mods)
        common_genes = sorted(set(r_mods['gene_name'].values) & set(mods_series.index))
        if len(common_genes) > 0:
            r_labels = r_mods.set_index('gene_name').loc[common_genes, 'module'].values
            py_labels = mods_series.loc[common_genes].values
            r_binary = (r_labels != 'grey').astype(int)
            unique_py = np.unique(py_labels)
            py_is_numeric = unique_py.dtype.kind in ('i', 'u', 'f')
            if py_is_numeric:
                py_binary = (py_labels > 0).astype(int)
            else:
                py_binary = (py_labels != 'grey').astype(int)
            try:
                from sklearn.metrics import adjusted_rand_score as _ari_fn
                from sklearn.metrics import normalized_mutual_info_score as _nmi_fn
                results['modules_ari'] = round(_ari_fn(r_labels.astype(str), py_labels.astype(str)), 4)
                results['modules_ari_binary'] = round(_ari_fn(r_binary, py_binary), 4)
                results['modules_nmi'] = round(_nmi_fn(r_labels.astype(str), py_labels.astype(str)), 4)
                results['modules_nmi_binary'] = round(_nmi_fn(r_binary, py_binary), 4)
            except ImportError:
                results['modules_ari'] = 'sklearn unavailable'
                results['modules_nmi'] = 'sklearn unavailable'
            results['common_genes'] = len(common_genes)
            results['r_modules'] = sorted(set(r_labels))
            results['py_modules'] = sorted([str(x) for x in unique_py])
        else:
            results['modules_ari'] = np.nan
            results['modules_ari_binary'] = np.nan
            results['common_genes'] = 0
            results['error'] = 'no overlapping genes'
    else:
        results['modules_ari'] = np.nan
        results['common_genes'] = 0
        results['error'] = 'no module colors/labels found'

    r_mes = r_outputs['mes']
    py_meson = hdW.get('MEs', None)
    if py_meson is None:
        py_meson = hdW.get('hMEs', None)
    if py_meson is None:
        py_meson = hdW.get('eigengenes', None)
    hme_results = {}
    if r_mes is not None and len(r_mes) > 0 and py_meson is not None:
        if isinstance(py_meson, pd.DataFrame):
            py_non_grey_cols = [c for c in py_meson.columns
                                 if str(c).lower() != 'grey' and str(c).lower() != '0']
        elif isinstance(py_meson, dict):
            py_non_grey_cols = [k.replace('ME_', '').replace('hME_', '')
                                for k in py_meson.keys()
                                if str(k).lower() != 'grey' and str(k).lower() != '0']
        else:
            py_non_grey_cols = []
        r_non_grey = [c for c in r_mes.columns if str(c).lower() != 'grey']
        n_pairs = min(len(r_non_grey), len(py_non_grey_cols))
        if n_pairs > 0:
            best_matches = []
            for r_mod_name in r_non_grey:
                r_vals = r_mes[r_mod_name].dropna()
                best_r, best_n, best_py = 0, 0, None
                for py_mod_name in py_non_grey_cols:
                    if isinstance(py_meson, pd.DataFrame):
                        py_vals = py_meson[py_mod_name].dropna().reset_index(drop=True)
                    elif isinstance(py_meson, dict):
                        py_raw = (py_meson.get(py_mod_name)
                                or py_meson.get(f'ME_{py_mod_name}')
                                or py_meson.get(f'hME_{py_mod_name}'))
                        if py_raw is None:
                            continue
                        py_vals = pd.Series(py_raw).dropna().reset_index(drop=True)
                    else:
                        continue
                    n_min = min(len(r_vals), len(py_vals))
                    if n_min < 3:
                        continue
                    try:
                        rv, pv = _pearsonr(r_vals.values[:n_min], py_vals.values[:n_min])
                        if abs(rv) > abs(best_r):
                            best_r, best_n, best_py = rv, n_min, str(py_mod_name)
                    except Exception:
                        continue
                if best_py is not None and abs(best_r) > 0:
                    hme_results[f'{r_mod_name}_vs_{best_py}'] = {
                        'r': round(best_r, 4), 'n': best_n,
                        'R_module': r_mod_name, 'Py_module': best_py,
                    }
            results['hmes'] = hme_results
            results['hmes_pairs'] = n_pairs
            results['hmes_common_cells'] = n_min if 'n_min' in dir() else 0
        else:
            results['hmes'] = {}
            results['hmes_error'] = f'no non-grey pairs: R={len(r_non_grey)} Py={len(py_non_grey_cols)}'
    else:
        results['hmes'] = {}
        missing = []
        if r_mes is None or len(r_mes) == 0:
            missing.append('R MEs')
        if py_meson is None:
            missing.append('Py MEs')
        results['hmes_error'] = f'missing: {", ".join(missing)}'

    r_kme = r_outputs['kme']
    py_kme_data = hdW.get('kME', None)
    kme_results = {}
    if r_kme is not None and len(r_kme) > 0 and py_kme_data is not None:
        r_kme_non_grey = [c for c in r_kme.columns if c.startswith('kME_') and 'grey' not in str(c).lower()]
        common_genes_kme = sorted(set(r_kme.index) & set(adata.var_names))
        py_mod_names = hdW.get('module_names', [])
        valid_mods = hdW.get('valid_modules', [])
        if isinstance(py_kme_data, np.ndarray):
            n_py_mods = py_kme_data.shape[1] if len(py_kme_data.shape) > 1 else 1
            py_kme_non_grey = list(range(n_py_mods))
            if len(py_mod_names) >= n_py_mods:
                py_kme_non_grey = [py_mod_names[j] for j in range(n_py_mods)]
            elif len(valid_mods) > 0:
                py_kme_non_grey = [str(m) for m in valid_mods]
        elif isinstance(py_kme_data, pd.DataFrame):
            py_kme_non_grey = [c for c in py_kme_data.columns
                               if str(c).lower() != 'grey' and str(c).lower() != '0']
        elif isinstance(py_kme_data, dict):
            py_kme_non_grey = [k.replace('kME_', '') for k in py_kme_data.keys()
                               if 'grey' not in str(k).lower()]
        else:
            py_kme_non_grey = []
        n_kme_pairs = min(len(r_kme_non_grey), len(py_kme_non_grey))
        if n_kme_pairs > 0 and len(common_genes_kme) >= 3:
            for r_col in r_kme_non_grey:
                matched = [g for g in common_genes_kme if g in r_kme.index and g in adata.var_names]
                if len(matched) < 3:
                    continue
                r_vals = r_kme.loc[matched, r_col].dropna()
                best_r, best_n, best_py = 0, 0, None
                for py_mod_id in py_kme_non_grey:
                    if isinstance(py_kme_data, np.ndarray):
                        kme_genes = hdW.get('dat_expr_genes', [])
                        if len(kme_genes) == 0:
                            kme_genes = adata.var_names[:py_kme_data.shape[0]]
                        gene_to_kme_idx = {g: i for i, g in enumerate(kme_genes)}
                        valid_pairs = [(g, gene_to_kme_idx[g]) for g in matched
                                       if g in gene_to_kme_idx]
                        if len(valid_pairs) < 3:
                            continue
                        matched_g, kme_idx = zip(*valid_pairs)
                        col_idx = py_kme_non_grey.index(py_mod_id) if isinstance(py_mod_id, str) else py_mod_id
                        if col_idx >= py_kme_data.shape[1]:
                            continue
                        py_arr = py_kme_data[list(kme_idx), col_idx]
                        py_vals = pd.Series(py_arr, index=list(matched_g)).dropna()
                    elif isinstance(py_kme_data, pd.DataFrame):
                        try:
                            py_vals = py_kme_data.loc[matched, py_mod_id].dropna()
                        except (KeyError, IndexError):
                            continue
                    elif isinstance(py_kme_data, dict):
                        py_raw = py_kme_data.get(py_mod_id) or py_kme_data.get(f'kME_{py_mod_id}')
                        if py_raw is None:
                            continue
                        py_vals = pd.Series(py_raw[:len(matched)], index=matched).dropna()
                    else:
                        continue
                    shared_idx = r_vals.index.intersection(py_vals.index)
                    if len(shared_idx) < 3:
                        continue
                    r_clean = pd.to_numeric(r_vals.loc[shared_idx], errors='coerce').dropna()
                    p_clean = pd.to_numeric(py_vals.loc[shared_idx], errors='coerce').dropna()
                    common = r_clean.index.intersection(p_clean.index)
                    if len(common) < 3:
                        continue
                    try:
                        rv, pv = _pearsonr(r_clean[common].values, p_clean[common].values)
                        if abs(rv) > abs(best_r):
                            best_r, best_n, best_py = rv, int(len(common)), str(py_mod_id)
                    except Exception:
                        continue
                if best_py is not None and abs(best_r) > 0:
                    kme_results[f'{r_col}_vs_{best_py}'] = {
                        'r': round(best_r, 4), 'n': best_n,
                        'R_col': r_col.replace('kME_', ''), 'Py_module': best_py,
                    }
            results['kme'] = kme_results
            results['kme_pairs'] = n_kme_pairs
            results['kme_common_genes'] = len(matched) if 'matched' in dir() else len(common_genes_kme)
        else:
            results['kme'] = {}
            results['kme_error'] = f'no pairs or too few: pairs={n_kme_pairs}, genes={len(common_genes_kme)}'
    else:
        results['kme'] = {}
        missing_k = []
        if r_kme is None:
            missing_k.append('R kME')
        if py_kme_data is None:
            missing_k.append('Py kME')
        results['kme_error'] = f'missing: {", ".join(missing_k)}'

    return results


def format_benchmark_report(results):
    lines = []
    lines.append("=" * 60)
    lines.append("  Benchmark: py-hdWGCNA vs R hdWGCNA")
    lines.append("=" * 60)

    pow_res = results.get('power', {})
    if 'r' in pow_res and not np.isnan(pow_res.get('r', np.nan)):
        lines.append(f"\n  Power table (SFT.R.sq):  r = {pow_res['r']:.4f}  (n={pow_res.get('n', '?')})")
    else:
        lines.append(f"\n  Power table:  ERROR - {pow_res.get('error', 'unknown')}")

    ari = results.get('modules_ari', np.nan)
    ari_bin = results.get('modules_ari_binary', np.nan)
    nmi = results.get('modules_nmi', np.nan)
    nmi_bin = results.get('modules_nmi_binary', np.nan)
    ng = results.get('common_genes', 0)
    r_mods = results.get('r_modules', [])
    py_mods = results.get('py_modules', [])
    lines.append(f"  Module assignment:")
    lines.append(f"    ARI       = {ari}  (binary: {ari_bin})")
    lines.append(f"    NMI       = {nmi}  (binary: {nmi_bin})")
    lines.append(f"    genes     = {ng}")
    lines.append(f"    R mods    = {r_mods}")
    lines.append(f"    Py mods   = {py_mods}")

    hmes = results.get('hmes', {})
    if hmes:
        lines.append("\n  Module eigengenes (hMEs):")
        for key, m in hmes.items():
            r_m = m.get('R_module', key)
            p_m = m.get('Py_module', key)
            lines.append(f"    {r_m:>12s} vs {p_m:<12s}:  r = {m['r']:.4f}  (n={m['n']})")
        hp = results.get('hmes_pairs', len(hmes))
        hc = results.get('hmes_common_cells', '?')
        lines.append(f"    {'':>12s}  pairs={hp}  cells ~ {hc}")
    else:
        err = results.get('hmes_error', '')
        if err:
            lines.append(f"\n  Module eigengenes (hMEs):  ERROR - {err}")

    kme = results.get('kme', {})
    if kme:
        lines.append("\n  kME (hub gene connectivity):")
        for key, m in kme.items():
            r_c = m.get('R_col', key)
            p_k = m.get('Py_key', key)
            lines.append(f"    {r_c:>12s} vs {p_k:<12s}:  r = {m['r']:.4f}  (n={m['n']})")
        kp = results.get('kme_pairs', len(kme))
        kg = results.get('kme_common_genes', '?')
        lines.append(f"    {'':>12s}  pairs={kp}  genes = {kg}")
    else:
        err = results.get('kme_error', '')
        if err:
            lines.append(f"\n  kME:  ERROR - {err}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
