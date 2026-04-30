"""
Network construction functions for py-hdWGCNA.
Core WGCNA engine: soft power testing and co-expression network construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def test_soft_powers(
    adata: AnnData,
    power_range: list = None,
    network_type: str = 'signed',
    cor_method: str = 'pearson',
    wgcna_name: str = None,
    **kwargs
):
    """
    Test different soft-thresholding powers for scale-free topology fit.
    
    Re-implements R hdWGCNA::TestSoftPowers (which calls WGCNA::pickSoftThreshold).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with datExpr stored
    power_range : list
        Range of powers to test (default: 1-20)
    network_type : str
        Network type: 'signed', 'unsigned', or 'signed hybrid'
    cor_method : str
        Correlation method: 'bicor', 'pearson', or 'spearman'
    wgcna_name : str
        Name of hdWGCNA experiment
    **kwargs
        Additional parameters passed to blockwiseModules
    
    Returns
    -------
    AnnData
        Modified AnnData with power table stored in hdWGCNA experiment
    
    Notes
    -----
    For each power value, computes:
    - SFT.R.sq: Scale-free topology fit index (R^2)
    - slope: Slope of log(k) vs log(p(k))
    - truncated.R.sq: Truncated R^2
    - mean.k., median.k., max.k.: Connectivity statistics
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data
    from .utils import compute_correlation_matrix, soft_threshold, scale_free_fit_index_full
    
    if power_range is None:
        power_range = list(range(1, 11)) + list(range(12, 31, 2))
    
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    
    if 'dat_expr' not in wgcna_data:
        raise ValueError("datExpr not found. Run SetDatExpr first.")
    
    dat_expr = wgcna_data['dat_expr']

    n_genes, n_samples = dat_expr.shape

    print(f"Testing {len(power_range)} soft powers on {n_genes} genes x {n_samples} samples...")

    t0 = __import__('time').time()
    if 'cor_matrix' in wgcna_data and wgcna_data['cor_matrix'] is not None:
        cor_matrix = wgcna_data['cor_matrix']
        print(f"  Using cached correlation matrix")
    else:
        cor_matrix = compute_correlation_matrix(dat_expr, method=cor_method)
        t1 = __import__('time').time()
        print(f"  {cor_method} correlation matrix computed in {t1-t0:.1f}s")

    if network_type == 'signed':
        cor_matrix += 1
        cor_matrix /= 2.0
        cor_transformed = cor_matrix
    elif network_type == 'unsigned':
        np.abs(cor_matrix, out=cor_matrix)
        cor_transformed = cor_matrix
    elif network_type == 'signed hybrid':
        np.clip(cor_matrix, 0, None, out=cor_matrix)
        cor_transformed = cor_matrix
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    np.fill_diagonal(cor_transformed, 0)

    np.clip(cor_transformed, 1e-300, None, out=cor_transformed)
    np.log(cor_transformed, out=cor_transformed)
    log_cor = cor_transformed

    results_list = []

    chunk_size = 100
    for power in power_range:
        k = np.zeros(n_genes, dtype=np.float64)
        for ci in range(0, n_genes, chunk_size):
            ci_end = min(ci + chunk_size, n_genes)
            chunk = log_cor[ci:ci_end, :]
            k[ci:ci_end] = np.exp(power * chunk).sum(axis=1)

        sft = scale_free_fit_index_full(k, nBreaks=10)
        
        results_list.append({
            'Power': power,
            'SFT.R.sq': sft['R2'],
            'slope': sft['slope'],
            'truncated.R.sq': sft['truncated_R2'],
            'mean.k.': float(np.mean(k)),
            'median.k.': float(np.median(k)),
            'max.k.': float(np.max(k))
        })
    
    power_table = pd.DataFrame(results_list)
    
    # R pickSoftThreshold strategy: find lowest power with SFT.R.sq >= 0.85
    RsquaredCut = 0.85
    selected_power = None
    
    # Iterate from small to large power, find first one meeting threshold
    for i, row in power_table.iterrows():
        if row['SFT.R.sq'] >= RsquaredCut:
            selected_power = int(row['Power'])
            break
    
    # Fallback: if no power meets threshold, use the one with highest SFT.R.sq
    if selected_power is None:
        best_power_idx = np.argmax(power_table['SFT.R.sq'].values)
        selected_power = int(power_table.loc[best_power_idx, 'Power'])
    
    wgcna_data['power_table'] = power_table
    wgcna_data['selected_power'] = selected_power
    wgcna_data['network_type'] = network_type
    wgcna_data['cor_method'] = cor_method
    wgcna_data['cor_matrix'] = cor_matrix
    
    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)
    
    print(f"TestSoftPowers complete: selected power = {selected_power}")
    
    return adata


def construct_network(
    adata: AnnData,
    power: int = None,
    tom_name: str = "hdWGCNA_TOM",
    network_type: str = 'signed',
    tom_type: str = "signed",
    tom_denom: str = "min",
    minModuleSize: int = 50,
    deepSplit: int = 4,
    pamRespectsDendro: bool = True,
    pamStage: bool = False,
    detectCutHeight: float = 0.995,
    minKMEtoStay: float = 0,
    mergeCutHeight: float = 0.2,
    n_threads: int = 1,
    verbose: int = 3,
    saveTOMs: bool = False,
    loadTOMs: bool = False,
    wgcna_name: str = None,
    **kwargs
):
    """
    Construct co-expression network using WGCNA approach.
    
    Re-implements R hdWGCNA::ConstructNetwork (which calls WGCNA::blockwiseModules).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with datExpr stored
    power : int
        Soft-thresholding power (auto-selected from TestSoftPowers if not provided)
    tom_name : str
        Name for TOM file storage
    network_type : str
        Network type: 'signed', 'unsigned', or 'signed hybrid'
    tom_type : str
        TOM type: 'unsigned' or 'signed'
    minModuleSize : int
        Minimum module size (number of genes)
    deepSplit : int
        Deep split sensitivity for dynamic tree cut (0-4, default 4 matching R)
    pamRespectsDendro : bool
        Whether PAM respects dendrogram structure
    detectCutHeight : str
        Detection cut height for module detection
    minKMEtoStay : float
        Minimum kME to stay in module after merging
    mergeCutHeight : float
        Cut height for module merging
    n_threads : int
        Number of threads for parallel computation
    verbose : int
        Verbosity level
    saveTOMs : bool
        Whether to save TOM to disk
    loadTOMs : bool
        Whether to load pre-computed TOM from disk
    wgcna_name : str
        Name of hdWGCNA experiment
    **kwargs
        Additional parameters passed to WGCNA functions
    
    Returns
    -------
    AnnData
        Modified AnnData with network results stored in hdWGCNA experiment
    
    Notes
    -----
    This function performs:
    1. Compute correlation matrix
    2. Apply soft-thresholding to create adjacency matrix
    3. Compute Topological Overlap Matrix (TOM)
    4. Hierarchical clustering on TOM dissimilarity
    5. Dynamic tree cut to identify modules
    6. Merge similar modules
    7. Compute module eigengenes
    """
    from .utils import (
        check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data,
        compute_correlation_matrix, soft_threshold, compute_tom,
        hclust_and_cut, compute_module_eigengenes, compute_kme
    )
    
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    
    if 'dat_expr' not in wgcna_data:
        raise ValueError("datExpr not found. Run SetDatExpr first.")
    
    dat_expr = wgcna_data['dat_expr']
    gene_names = wgcna_data.get('dat_expr_genes', [f"Gene_{i}" for i in range(dat_expr.shape[0])])

    n_genes, n_samples = dat_expr.shape
    
    if power is None:
        if 'selected_power' in wgcna_data:
            power = wgcna_data['selected_power']
        else:
            power = 6
    
    print(f"ConstructNetwork: building co-expression network with power={power}, "
          f"{n_genes} genes, {n_samples} samples")
    
    cor_method = wgcna_data.get('cor_method', 'pearson')
    print(f"Computing correlation matrix ({cor_method})...")
    cor_matrix = compute_correlation_matrix(dat_expr, method=cor_method)
    np.clip(cor_matrix, -1, 1, out=cor_matrix)
    print(f"  Cor matrix shape: {cor_matrix.shape}, range=[{cor_matrix.min():.4f}, {cor_matrix.max():.4f}]")

    print("Applying soft-thresholding...")
    adj_matrix = soft_threshold(cor_matrix, power, network_type)
    print(f"  Adj matrix shape: {adj_matrix.shape}, range=[{adj_matrix.min():.6f}, {adj_matrix.max():.6f}]")

    del cor_matrix

    print("Computing Topological Overlap Matrix (TOM)...")
    tom_dissim = compute_tom(adj_matrix, tom_type=tom_type, tom_denom=tom_denom)
    print(f"  TOM dissim shape: {tom_dissim.shape}, range=[{tom_dissim.min():.4f}, {tom_dissim.max():.4f}]")
    
    del adj_matrix
    
    print("Performing hierarchical clustering + dynamic tree cutting in R...")
    labels = hclust_and_cut(
        tom_dissim,
        deepSplit=deepSplit,
        pamRespectsDendro=pamRespectsDendro,
        pamStage=pamStage,
        minClusterSize=minModuleSize,
        cutHeight=detectCutHeight,
        method='average'
    )
    
    unique_labels = np.unique(labels)
    n_modules = len(unique_labels)
    non_grey = len([l for l in unique_labels if l != 0])
    
    if non_grey == 0 and detectCutHeight is not None:
        print(f"  detectCutHeight={detectCutHeight} produced 0 modules, retrying with auto cutHeight...")
        labels = hclust_and_cut(
            tom_dissim,
            deepSplit=deepSplit,
            pamRespectsDendro=pamRespectsDendro,
            pamStage=pamStage,
            minClusterSize=minModuleSize,
            cutHeight=None,
            method='average'
        )
    
    unique_labels = np.unique(labels)
    n_modules = len(unique_labels)
    
    label_counts = pd.Series(labels).value_counts()
    grey_count = label_counts.get(0, 0)
    
    print(f"Identified {n_modules} modules ({grey_count} unassigned/grey)")
    
    MEs, var_explained, valid_mods = compute_module_eigengenes(dat_expr, labels)
    
    if len(valid_mods) > 1:
        print("Merging similar modules...")
        MEs_merged, labels_merged, valid_mods = _merge_close_modules(
            MEs, labels, cut_height=mergeCutHeight, dat_expr=dat_expr
        )
    else:
        MEs_merged = MEs
        labels_merged = labels
    
    final_unique_labels = np.unique(labels_merged)
    
    module_colors = _generate_module_colors(final_unique_labels)
    
    label_to_color = {}
    for lbl in final_unique_labels:
        if lbl == 0:
            label_to_color[lbl] = 'grey'
        else:
            label_to_color[lbl] = module_colors.get(lbl, 'grey')
    
    colors_array = np.array([label_to_color.get(l, 'grey') for l in labels_merged])
    
    modules_df = pd.DataFrame({
        'gene_name': gene_names,
        'module': [f"M{l}" if l != 0 else "grey" for l in labels_merged],
        'color': colors_array
    })
    
    kME_all = compute_kme(dat_expr, MEs_merged, method=cor_method)

    kME_cols = {}

    n_merged_modules = MEs_merged.shape[0]
    merged_mod_names = [f"M{m}" for m in valid_mods]

    for j, mod_name in enumerate(merged_mod_names):
        col_name = f"kME_{mod_name}"
        if j < kME_all.shape[1]:
            kME_cols[col_name] = kME_all[:, j]
    
    for col_name, kME_vals in kME_cols.items():
        modules_df[col_name] = kME_vals
    
    TOM_sim = 1.0 - tom_dissim
    np.fill_diagonal(TOM_sim, 1.0)

    network_data = {
        'tom_dissim': tom_dissim,
        'TOM': TOM_sim,
        'module_labels': labels_merged,
        'modules_df': modules_df,
        'MEs': MEs_merged,
        'hMEs': MEs_merged.copy(),
        'kME': kME_all,
        'var_explained': var_explained,
        'valid_modules': valid_mods,
        'network_params': {
            'power': power,
            'network_type': network_type,
            'tom_type': tom_type,
            'minModuleSize': minModuleSize,
            'deepSplit': deepSplit,
            'mergeCutHeight': mergeCutHeight
        },
        'tom_name': tom_name
    }
    
    wgcna_data.update(network_data)
    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)
    
    n_final_modules = len([m for m in final_unique_labels if m != 0])
    print(f"ConstructNetwork complete: {n_final_modules} co-expression modules identified")
    
    return adata


def _merge_close_modules(MEs: np.ndarray, labels: np.ndarray, 
                          cut_height: float = 0.25,
                          dat_expr: np.ndarray = None) -> tuple:
    n_modules = MEs.shape[0]
    
    if n_modules <= 1:
        return MEs, labels, list(set(labels) - {0})
    
    return _merge_via_python(MEs, labels, cut_height, dat_expr)


def _merge_via_python(MEs: np.ndarray, labels: np.ndarray,
                       cut_height: float, dat_expr: np.ndarray = None) -> tuple:
    n_modules = MEs.shape[0]
    
    if n_modules <= 1:
        return MEs, labels, list(set(labels) - {0})
    
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    me_cor = np.corrcoef(MEs)
    me_cor = np.nan_to_num(me_cor, nan=0.0)

    me_dissim = 1 - me_cor
    np.fill_diagonal(me_dissim, 0)

    print(f"  Merge step: n_modules={MEs.shape[0]}, cut_height={cut_height}")
    print(f"  ME correlation range: [{me_cor.min():.4f}, {me_cor.max():.4f}]")
    print(f"  ME dissimilarity range: [{me_dissim[me_dissim > 0].min():.4f}, {me_dissim.max():.4f}]")
    for i in range(me_cor.shape[0]):
        for j in range(i+1, me_cor.shape[0]):
            print(f"    Module {i+1} vs Module {j+1}: cor={me_cor[i,j]:.4f}, dissim={me_dissim[i,j]:.4f}")

    Z_merge = linkage(squareform(me_dissim, checks=False), method='average')
    merged_labels = fcluster(Z_merge, t=cut_height, criterion='distance')
    print(f"  fcluster labels: {merged_labels}")
    
    merged_labels = fcluster(Z_merge, t=cut_height, criterion='distance')

    def _first_appearance_unique(arr):
        seen = set()
        result = []
        for v in arr:
            if v not in seen:
                result.append(v)
                seen.add(v)
        return result

    _fa_labels = _first_appearance_unique(labels)

    new_labels = labels.copy()
    unique_old = [m for m in _fa_labels if m != 0]

    label_to_me_index = {lbl: i for i, lbl in enumerate(_fa_labels) if lbl != 0}
    old_to_new = {}

    new_id = 1
    for old_id in unique_old:
        if old_id == 0:
            continue

        old_indices = np.where(labels == old_id)[0]

        if len(old_indices) == 0:
            continue

        me_index = label_to_me_index.get(old_id)

        if me_index is None or me_index >= len(merged_labels):
            continue

        target_cluster = merged_labels[me_index]

        min_old_in_cluster = float('inf')
        for other_old in unique_old:
            if other_old == 0:
                continue
            other_me_idx = label_to_me_index.get(other_old)
            if other_me_idx is not None and other_me_idx < len(merged_labels):
                if merged_labels[other_me_idx] == target_cluster and other_old < min_old_in_cluster:
                    min_old_in_cluster = other_old

        old_to_new[old_id] = min_old_in_cluster
    
    for old_id, new_id_map in old_to_new.items():
        new_labels[labels == old_id] = new_id_map
    
    final_MEs = []
    final_valid_mods = []

    _fa_new = _first_appearance_unique(new_labels)
    unique_new_labels = [m for m in _fa_new if m != 0]

    for mod_id in unique_new_labels:
        if dat_expr is not None:
            gene_indices = np.where(new_labels == mod_id)[0]
            if len(gene_indices) > 0:
                module_expr = dat_expr[gene_indices, :]
                if module_expr.shape[0] > 1:
                    mod_centered = module_expr - np.mean(module_expr, axis=1, keepdims=True)
                    mod_std = np.std(module_expr, axis=1, ddof=1, keepdims=True)
                    mod_std = np.where(mod_std < 1e-10, 1.0, mod_std)
                    mod_scaled = mod_centered / mod_std
                    from sklearn.decomposition import PCA as SklearnPCA
                    pca = SklearnPCA(n_components=1)
                    me = pca.fit_transform(mod_scaled.T).flatten()
                    mean_expr = module_expr.mean(axis=0)
                    if np.corrcoef(me, mean_expr)[0, 1] < 0:
                        me = -me
                else:
                    me = module_expr[0].copy()
                final_MEs.append(me)
                final_valid_mods.append(mod_id)
        else:
            old_modules_in_cluster = [old_id for old_id, new_id_map in old_to_new.items() if new_id_map == mod_id]
            me_indices_for_this_cluster = [label_to_me_index[old_id] for old_id in old_modules_in_cluster if old_id in label_to_me_index]
            if len(me_indices_for_this_cluster) > 0:
                cluster_ME = MEs[me_indices_for_this_cluster].mean(axis=0)
                final_MEs.append(cluster_ME)
                final_valid_mods.append(mod_id)

    if len(final_MEs) > 0:
        final_MEs = np.array(final_MEs)
    else:
        final_MEs = MEs
        final_valid_mods = [m for m in _fa_labels if m != 0]
    
    return final_MEs, new_labels, final_valid_mods


def _generate_module_colors(unique_labels: np.ndarray) -> dict:
    """
    Generate distinct colors for each module.
    
    Uses standard WGCNA color palette.
    
    Parameters
    ----------
    unique_labels : np.ndarray
        Unique module labels
    
    Returns
    -------
    dict
        Mapping from label to color name
    """
    wgcna_colors = [
        '#4FC3F7', '#1E88E5', '#7B1FA2', '#FDD835', '#43A047',
        '#E53935', '#212121', '#F48FB1', '#AB47BC', '#8E24AA',
        '#9CCC65', '#D7CCC8', '#26C6DA', '#FF8A65', '#81D4FA',
        '#FFB300', '#EA80FC', '#FF7043', '#1A237E', '#B2EBF2',
        '#8D6E63', '#689F38', '#FF5722', '#BA68C8', '#FFFFEE',
        '#FFAB91', '#80CBC4', '#B0BEC5', '#FFECB3',
        '#C8E6C9', '#8BC34A', '#00ACC1', '#FFF59D', '#B71C1C',
        '#8E24AA', '#8D6E63', '#C62828', '#7B1FA2', '#B3E5FC',
        '#66BB6A', '#2E7D32', '#78909C', '#FF8F00', '#E57373',
        '#7E57C2', '#AD1457', '#D81B60', '#FFCDD2', '#D7CCC8',
        '#FFF59D', '#80DEEA', '#B2FF59', '#C5E1A5', '#A1887F',
        '#EF5350', '#4DB6AC', '#BCAAA4', '#FFAB76', '#F8BBD9'
    ]
    wgcna_color_names = [
        'turquoise', 'blue', 'brown', 'yellow', 'green',
        'red', 'black', 'pink', 'magenta', 'purple',
        'greenyellow', 'tan', 'cyan', 'salmon', 'skyblue',
        'gold', 'violet', 'orange', 'midnightblue', 'lightcyan',
        'saddlebrown', 'darkolivegreen', 'coral', 'orchid', 'ivory',
        'darksalmon', 'aquamarine', 'lightsteelblue', 'navajowhite',
        'beige', 'limegreen', 'darkturquoise', 'khaki', 'darkred',
        'darkmagenta', 'chocolate', 'crimson', 'darkviolet', 'powderblue',
        'seagreen', 'forestgreen', 'slategray', 'darkgoldenrod', 'indianred',
        'mediumpurple', 'thistle', 'plum', 'mistyrose', 'darkblue',
        'lightpink', 'burlywood', 'lightgoldenrod', 'paleturquoise', 'springgreen',
        'olivedrab', 'sienna', 'lightskyblue', 'peachpuff', 'steelblue',
        'firebrick', 'cadetblue', 'rosybrown', 'chocolate1', 'palevioletred'
    ]
    wgcna_hex_map = dict(zip(wgcna_color_names, wgcna_colors))

    color_map = {}
    color_idx = 0

    sorted_labels = sorted([l for l in unique_labels if l != 0])

    for lbl in sorted_labels:
        if color_idx < len(wgcna_color_names):
            color_name = wgcna_color_names[color_idx]
            color_map[lbl] = wgcna_hex_map.get(color_name, color_name)
        else:
            color_map[lbl] = '#888888'
        color_idx += 1
    
    return color_map


def get_tom(adata: AnnData, wgcna_name: str = None) -> np.ndarray:
    """
    Retrieve the Topological Overlap Matrix (similarity).
    
    Re-implements R hdWGCNA::GetTOM behavior.
    Returns the TOM similarity matrix (not dissimilarity).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with hdWGCNA results
    wgcna_name : str
        Name of hdWGCNA experiment
    
    Returns
    -------
    np.ndarray
        TOM similarity matrix
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data
    
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    
    if 'TOM' in wgcna_data:
        return wgcna_data['TOM']
    
    if 'tom_dissim' in wgcna_data:
        TOM_sim = 1.0 - wgcna_data['tom_dissim']
        np.fill_diagonal(TOM_sim, 1.0)
        return TOM_sim
    
    raise ValueError("TOM not found. Run ConstructNetwork first.")


def get_power_table(adata: AnnData, wgcna_name: str = None) -> pd.DataFrame:
    """
    Retrieve the soft power test results table.
    
    Re-implements R hdWGCNA::GetPowerTable behavior.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with hdWGCNA results
    wgcna_name : str
        Name of hdWGCNA experiment
    
    Returns
    -------
    pd.DataFrame
        Power test results table
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data
    
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    
    if 'power_table' not in wgcna_data:
        raise ValueError("Power table not found. Run TestSoftPowers first.")
    
    return wgcna_data['power_table']
