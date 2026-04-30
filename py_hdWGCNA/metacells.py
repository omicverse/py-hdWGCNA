"""
Metacell construction functions for py-hdWGCNA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors


def metacells_by_groups(
    adata: AnnData,
    group_by: list | str = None,
    reduction: str = 'pca',
    k: int = 25,
    max_shared: int = 15,
    ident_group: str = None,
    min_cells: int = 100,
    target_metacells: int = 1000,
    max_iter: int = 5000,
    wgcna_name: str = "hdWGCNA"
):
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data
    
    if isinstance(group_by, str):
        group_by = [group_by]
    
    if group_by is None:
        raise ValueError("group_by must be specified")
    
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    
    if reduction not in adata.obsm:
        available_reductions = list(adata.obsm.keys())
        raise ValueError(f"Reduction '{reduction}' not found. Available: {available_reductions}")
    
    coords = adata.obsm[reduction]
    
    obs_df = adata.obs.copy()
    
    obs_df['__groups__'] = obs_df[group_by].apply(
        lambda x: '#'.join(x.astype(str)), axis=1
    )
    
    all_metacell_exprs = []
    all_metacell_obs = []
    metacell_to_cell_map = {}
    
    unique_groups = obs_df['__groups__'].unique()
    
    group_counts = obs_df['__groups__'].value_counts()
    skipped = [g for g in unique_groups if group_counts.get(g, 0) < min_cells]
    if skipped:
        print(f"Removing the following groups that did not meet min_cells: {', '.join(skipped)}")
    unique_groups = [g for g in unique_groups if group_counts.get(g, 0) >= min_cells]
    
    for group_val in unique_groups:
        
        group_mask = obs_df['__groups__'] == group_val
        group_idx = np.where(group_mask)[0]
        
        n_cells_in_group = len(group_idx)
        
        print(f"Processing group '{group_val}': {n_cells_in_group} cells")
        
        group_coords = coords[group_idx].copy()
        
        if n_cells_in_group <= k:
            print(f"  Only {n_cells_in_group} cells (<= k={k}), creating single metacell")
            chosen = list(range(n_cells_in_group))
        else:
            chosen = _bootstrap_metacells(group_coords, k, max_shared, target_metacells, max_iter)
        
        if len(chosen) <= 1:
            print(f"  Metacell failed for group '{group_val}'")
            continue
        
        nn_map = _compute_knn_index(group_coords, k)
        
        cell_sample = nn_map[chosen, :]
        
        if 'counts' in adata.layers:
            counts_mat = adata.layers['counts']
        else:
            counts_mat = adata.X
        if hasattr(counts_mat, 'toarray'):
            counts_mat = counts_mat.toarray()
        counts_mat = np.array(counts_mat, dtype=np.float64)
        
        n_metacells = len(chosen)
        
        mc_counts = np.zeros((n_metacells, counts_mat.shape[1]), dtype=np.float64)
        
        mc_expr_list = []
        mc_obs_list = []
        
        for mc_i in range(n_metacells):
            members_local = cell_sample[mc_i, :]
            members_local = np.unique(members_local)
            original_cell_indices = group_idx[members_local]
            
            mc_counts[mc_i, :] = counts_mat[original_cell_indices, :].sum(axis=0) / k
            
            mc_expr_list.append(mc_counts[mc_i, :])
            
            obs_row = {
                '__groups__': group_val,
                'metacell_id': mc_i,
                'n_cells': len(members_local),
            }
            
            for gb in group_by:
                idx_labels = obs_df.index[group_idx[members_local]]
                col_vals = obs_df.loc[idx_labels, gb].values
                obs_row[gb] = str(col_vals[0]) if len(col_vals) > 0 else ''
            
            mc_obs_list.append(obs_row)
            
            metacell_name = f"{group_val}_{mc_i + 1}"
            metacell_to_cell_map[metacell_name] = original_cell_indices.tolist()
        
        if len(mc_expr_list) > 0:
            all_metacell_exprs.append(np.vstack(mc_expr_list))
            all_metacell_obs.extend(mc_obs_list)
    
    if len(all_metacell_exprs) == 0:
        raise ValueError("No metacells were created. Check min_cells parameter.")
    
    final_counts = np.vstack(all_metacell_exprs)
    final_obs = pd.DataFrame(all_metacell_obs)

    mc_adata = AnnData(
        X=final_counts.copy(),
        var=pd.DataFrame(index=adata.var_names),
        obs=final_obs
    )

    mc_adata.layers['counts'] = final_counts.copy()

    mc_adata = _normalize_metacells(mc_adata)

    if ident_group and ident_group in mc_adata.obs.columns:
        mc_adata.obs[ident_group] = mc_adata.obs[ident_group].astype('category')
    
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name) if 'hdWGCNA' in adata.uns else {}
    wgcna_data['active_wgcna'] = wgcna_name
    wgcna_data['metacell_obj'] = mc_adata
    wgcna_data['metacell_groupby'] = group_by
    wgcna_data['metacell_k'] = k
    wgcna_data['metacell_max_shared'] = max_shared
    wgcna_data['metacell_reduction'] = reduction
    wgcna_data['metacell_to_cell_map'] = metacell_to_cell_map
    
    adata.uns['hdWGCNA'][wgcna_name] = wgcna_data
    adata.uns['hdWGCNA']['active_wgcna'] = wgcna_name
    
    total_mcs = mc_adata.n_obs
    print(f"MetacellsByGroups complete: {total_mcs} metacells created from {adata.n_obs} cells")
    
    return adata


def _normalize_metacells(mc_adata: AnnData, scale_factor: float = 10000.0) -> AnnData:
    counts = mc_adata.layers['counts'].copy()
    if hasattr(counts, 'toarray'):
        counts = counts.toarray()
    counts = np.array(counts, dtype=np.float64)
    lib_sizes = counts.sum(axis=1, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1.0
    norm_expr = (counts / lib_sizes) * scale_factor
    norm_expr = np.log1p(norm_expr)
    mc_adata.X = norm_expr.astype(np.float32)
    return mc_adata


def normalize_metacells(adata: AnnData, wgcna_name: str = None) -> AnnData:
    from .utils import check_wgcna_name, get_hdWGCNA_data
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    if 'metacell_obj' not in wgcna_data:
        raise ValueError("No metacell object found. Run MetacellsByGroups first.")
    mc_adata = wgcna_data['metacell_obj']
    if 'counts' in mc_adata.layers:
        mc_adata = _normalize_metacells(mc_adata)
        wgcna_data['metacell_obj'] = mc_adata
        adata.uns['hdWGCNA'][wgcna_name] = wgcna_data
        print("NormalizeMetacells complete")
    else:
        print("WARNING: No 'counts' layer found in metacell object, skipping normalization")
    return adata


def _compute_knn_index(coords: np.ndarray, k: int) -> np.ndarray:
    n_cells = coords.shape[0]
    n_neighbors = min(k, n_cells)

    if n_neighbors < 2:
        nn_map = np.arange(n_cells).reshape(-1, 1)
        return nn_map

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean', algorithm='auto')
    knn.fit(coords)
    _, indices = knn.kneighbors(coords)

    indices = indices[:, 1:]

    self_col = np.arange(n_cells).reshape(-1, 1)
    nn_map = np.hstack([indices, self_col])

    return nn_map


def _bootstrap_metacells(coords: np.ndarray, k: int, max_shared: int,
                          target_metacells: int, max_iter: int) -> list:
    n_cells = coords.shape[0]

    nn_map = _compute_knn_index(coords, k)

    good_choices = np.arange(n_cells, dtype=np.int64)
    good_mask = np.ones(n_cells, dtype=bool)
    chosen = []
    it = 0
    k2 = k * 2

    rng = np.random.default_rng()

    chosen_cell_sets = []

    while good_mask.any() and len(chosen) < target_metacells and it < max_iter:
        it += 1

        available = np.where(good_mask)[0]
        choice_idx = rng.integers(0, len(available))
        new_center = available[choice_idx]

        new_cell_set = set(nn_map[new_center, :].tolist())

        if len(chosen) == 0:
            chosen.append(int(new_center))
            chosen_cell_sets.append(new_cell_set)
            good_mask[new_center] = False
            continue

        max_shared_found = 0
        for existing_set in chosen_cell_sets:
            shared = k2 - len(existing_set | new_cell_set)
            if shared > max_shared_found:
                max_shared_found = shared
                if max_shared_found > max_shared:
                    break

        if max_shared_found <= max_shared:
            chosen.append(int(new_center))
            chosen_cell_sets.append(new_cell_set)

        good_mask[new_center] = False

    return chosen


def aggregate_gene_expression(
    adata: AnnData,
    group_by: str | list = None,
    assay: str = 'RNA',
    layer: str = 'data',
    method: str = 'mean',
    wgcna_name: str = None
):
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data
    
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    
    if isinstance(group_by, str):
        group_by = [group_by]
    
    expr_mat = _get_expr_matrix(adata, layer)
    
    obs_df = adata.obs.copy()
    groups_key = '__agg_groups__'
    obs_df[groups_key] = obs_df[group_by].apply(
        lambda x: '_'.join(x.astype(str)), axis=1
    )
    
    unique_groups = obs_df[groups_key].unique()
    
    agg_expr_list = []
    agg_obs_list = []
    
    for grp in unique_groups:
        mask = obs_df[groups_key] == grp
        idx = obs_df[mask].index
        
        group_expr = expr_mat[idx, :]
        
        if method == 'mean':
            agg_expr = group_expr.mean(axis=0)
        elif method == 'sum':
            agg_expr = group_expr.sum(axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        agg_expr_list.append(np.array(agg_expr).flatten())
        
        obs_row = {gb: obs_df.loc[idx[0], gb] for gb in group_by}
        obs_row['n_original_cells'] = len(idx)
        agg_obs_list.append(obs_row)
    
    pb_adata = AnnData(
        X=np.array(agg_expr_list),
        var=pd.DataFrame(index=adata.var_names),
        obs=pd.DataFrame(agg_obs_list)
    )
    
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    wgcna_data['pseudobulk_obj'] = pb_adata
    wgcna_data['pseudobulk_method'] = method
    wgcna_data['pseudobulk_groupby'] = group_by
    
    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)
    
    print(f"AggregateGeneExpression complete: {pb_adata.n_obs} pseudobulk samples")
    
    return adata


def _get_expr_matrix(adata: AnnData, layer: str = 'data') -> np.ndarray:
    if layer == 'X':
        mat = adata.X
    elif layer == 'data':
        mat = adata.raw.X if adata.raw is not None else adata.X
    elif layer in adata.layers:
        mat = adata.layers[layer]
    else:
        mat = adata.X
    
    if hasattr(mat, 'toarray'):
        mat = mat.toarray()
    
    return np.array(mat)


def get_metacell_object(adata: AnnData, wgcna_name: str = None) -> AnnData:
    from .utils import check_wgcna_name, get_hdWGCNA_data
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    if 'metacell_obj' not in wgcna_data:
        raise ValueError("No metacell object found. Run MetacellsByGroups first.")
    return wgcna_data['metacell_obj']


construct_metacells = metacells_by_groups
