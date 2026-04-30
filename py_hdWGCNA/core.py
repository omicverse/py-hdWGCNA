"""
Core functions for py-hdWGCNA: SetupForWGCNA and SetDatExpr.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def setup_for_wgcna(
    adata: AnnData,
    gene_select: str = "fraction",
    fraction: float = 0.05,
    n_genes: int = None,
    genes_use: list = None,
    wgcna_name: str = "hdWGCNA"
):
    """
    Set up AnnData object for hdWGCNA analysis.
    
    Re-implements R hdWGCNA::SetupForWGCNA behavior.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells x genes)
    gene_select : str
        Gene selection method: 'fraction', 'variable', or 'custom'
    fraction : float
        Fraction of cells a gene must be expressed in (for 'fraction' method)
    n_genes : int
        Number of top variable genes to select (for 'variable' method)
    genes_use : list
        Custom list of genes to use (for 'custom' method)
    wgcna_name : str
        Name for this hdWGCNA experiment
    
    Returns
    -------
    AnnData
        Modified AnnData with hdWGCNA setup data in .uns['hdWGCNA']
    
    Notes
    -----
    This function:
    1. Selects genes based on specified criteria
    2. Stores selected genes in hdWGCNA experiment metadata
    3. Initializes hdWGCNA experiment storage structure
    """
    
    if 'hdWGCNA' not in adata.uns:
        adata.uns['hdWGCNA'] = {}
    
    if gene_select == "fraction":
        
        if 'counts' in adata.layers:
            expr_mat = adata.layers['counts'].copy()
        elif adata.raw is not None:
            expr_mat = adata.raw.X.copy()
        else:
            expr_mat = adata.X.copy()
        
        if hasattr(expr_mat, 'toarray'):
            expr_mat = expr_mat.toarray()
        
        frac_cells_expr = np.array(
            (expr_mat > 0).sum(axis=0) / expr_mat.shape[0]
        ).flatten()
        
        genes_selected_mask = frac_cells_expr >= fraction
        
        genes_use = adata.var_names[genes_selected_mask].tolist()
        
    elif gene_select == "variable":
        if n_genes is None:
            n_genes = min(3000, adata.n_vars)
        
        if 'highly_variable' in adata.var.columns:
            hv_genes = adata.var[adata.var['highly_variable']].index.tolist()
            genes_use = hv_genes[:n_genes]
        else:
            mean_expr = np.array(adata.X.mean(axis=0)).flatten()
            var_expr = np.array(np.var(adata.X, axis=0)).flatten()
            
            cv2 = var_expr / (mean_expr ** 2 + 1e-8)
            top_idx = np.argsort(cv2)[::-1][:n_genes]
            genes_use = adata.var_names[top_idx].tolist()
    
    elif gene_select == "custom":
        if genes_use is None:
            raise ValueError("genes_use must be provided when gene_select='custom'")
        
        genes_use = [g for g in genes_use if g in adata.var_names]
    else:
        raise ValueError(f"Unknown gene_select method: {gene_select}")
    
    wgcna_data = {
        'active_wgcna': wgcna_name,
        'gene_select_method': gene_select,
        'genes_use': genes_use,
        'n_genes': len(genes_use),
        'setup_complete': True
    }

    adata.uns['hdWGCNA'][wgcna_name] = wgcna_data
    adata.uns['hdWGCNA']['active_wgcna'] = wgcna_name

    print(f"SetupForWGCNA complete: {len(genes_use)} genes selected using '{gene_select}' method")
    adata = set_dat_expr(adata, group_by=None, group_name=None, layer='data', use_metacells=False, wgcna_name=wgcna_name)

    return adata


def set_dat_expr(
    adata: AnnData,
    group_name: str | list = None,
    group_by: str = None,
    assay: str = 'RNA',
    layer: str = 'data',
    use_metacells: bool = True,
    wgcna_name: str = None
):
    """
    Set up expression matrix (datExpr) for network construction.
    
    Re-implements R hdWGCNA::SetDatExpr behavior.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    group_name : str or list
        Name(s) of cell groups to include
    group_by : str
        Column name in adata.obs containing group labels
    assay : str
        Assay name (for Seurat compatibility, ignored in Python)
    layer : str
        Layer name to use for expression data
    use_metacells : bool
        Whether to use metacell expression matrix
    wgcna_name : str
        Name of hdWGCNA experiment
    
    Returns
    -------
    AnnData
        Modified AnnData with datExpr stored in hdWGCNA experiment
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data
    
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    
    if use_metacells and 'metacell_obj' in wgcna_data:
        mc_adata = wgcna_data['metacell_obj']
        
        subset_col = group_by
        if subset_col is not None and subset_col not in mc_adata.obs.columns:
            mc_groupby_list = wgcna_data.get('metacell_groupby', [])
            if isinstance(mc_groupby_list, list):
                for col in mc_groupby_list:
                    if col in mc_adata.obs.columns:
                        subset_col = col
                        break
        
        if subset_col is not None and subset_col in mc_adata.obs.columns and group_name is not None:
            if isinstance(group_name, str):
                cell_mask = mc_adata.obs[subset_col] == group_name
            else:
                cell_mask = mc_adata.obs[subset_col].isin(group_name)
            
            n_matching = cell_mask.sum()
            if n_matching > 0:
                mc_subset = mc_adata[cell_mask].copy()
                
                dat_expr = _get_expression_matrix(mc_subset, layer=layer)
                genes_use = mc_subset.var_names.tolist()
                cells_use = mc_subset.obs_names.tolist()
                print(f"  Subset metacells by {subset_col}={group_name}: {n_matching}/{mc_adata.n_obs} metacells")
            else:
                print(f"  WARNING: No metacells found for {subset_col}={group_name}, using all {mc_adata.n_obs} metacells")
                dat_expr = _get_expression_matrix(mc_adata, layer=layer)
                genes_use = mc_adata.var_names.tolist()
                cells_use = mc_adata.obs_names.tolist()
        else:
            dat_expr = _get_expression_matrix(mc_adata, layer=layer)
            genes_use = mc_adata.var_names.tolist()
            cells_use = mc_adata.obs_names.tolist()
    
    else:
        
        if group_by is not None and group_by in adata.obs.columns and group_name is not None:
            if isinstance(group_name, str):
                cell_mask = adata.obs[group_by] == group_name
            else:
                cell_mask = adata.obs[group_by].isin(group_name)
            
            adata_subset = adata[cell_mask].copy()
            
            dat_expr = _get_expression_matrix(adata_subset, layer=layer)
            genes_use = adata_subset.var_names.tolist()
            cells_use = adata_subset.obs_names.tolist()
        else:
            dat_expr = _get_expression_matrix(adata, layer=layer)
            genes_use = adata.var_names.tolist()
            cells_use = adata.obs_names.tolist()
    
    wgcna_genes = wgcna_data.get('genes_use', [])
    if len(wgcna_genes) > 0:
        wgcna_gene_set = set(wgcna_genes)
        gene_mask = [g in wgcna_gene_set for g in genes_use]
        gene_mask_arr = np.array(gene_mask)
        dat_expr = dat_expr[gene_mask_arr, :]
        genes_use = [g for g, m in zip(genes_use, gene_mask) if m]
    
    gene_vars = np.var(dat_expr, axis=1, ddof=1)
    good_mask = gene_vars > 1e-15
    nan_frac = np.isnan(dat_expr).sum(axis=1) / dat_expr.shape[1]
    good_mask &= nan_frac < 0.5
    if good_mask.sum() < 2:
        raise ValueError("Too few genes remaining after goodGenes check.")
    dat_expr = dat_expr[good_mask, :]
    genes_use = [g for g, m in zip(genes_use, good_mask) if m]
    print(f"  goodGenes filter: kept {good_mask.sum()}/{len(good_mask)} genes")
    
    wgcna_data['dat_expr'] = dat_expr
    wgcna_data['dat_expr_genes'] = genes_use
    wgcna_data['dat_expr_cells'] = cells_use
    wgcna_data['group_name'] = group_name
    wgcna_data['group_by'] = group_by
    
    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)
    
    print(f"SetDatExpr complete: {dat_expr.shape[0]} genes x {dat_expr.shape[1]} samples/cells")
    
    return adata


def _get_expression_matrix(adata: AnnData, layer: str = 'data') -> np.ndarray:
    """
    Extract expression matrix from AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object
    layer : str
        Layer name ('data', 'counts', 'scale', etc.)
    
    Returns
    -------
    np.ndarray
        Genes x Samples expression matrix
    """
    if layer == 'X':
        mat = adata.X
    elif layer == 'data':
        if adata.raw is not None:
            mat = adata.raw.X
        else:
            mat = adata.X
    elif layer in adata.layers:
        mat = adata.layers[layer]
    else:
        mat = adata.X
    
    if hasattr(mat, 'toarray'):
        mat = mat.toarray()
    
    return np.array(mat.T)


def normalize_metacells(adata: AnnData, wgcna_name: str = None) -> AnnData:
    """
    Normalize metacell expression matrix.
    
    Re-implements R hdWGCNA::NormalizeMetacells behavior.
    
    Uses log normalization: log1p(CPM * scale_factor).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with metacell object stored
    wgcna_name : str
        Name of hdWGCNA experiment
    
    Returns
    -------
    AnnData
        Modified AnnData with normalized metacell expression
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data
    
    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    
    if 'metacell_obj' not in wgcna_data:
        raise ValueError("No metacell object found. Run MetacellsByGroups first.")
    
    mc_adata = wgcna_data['metacell_obj']
    
    total_counts_per_cell = np.array(mc_adata.X.sum(axis=1)).flatten()
    scale_factor = 10000.0
    scale_factors = np.where(total_counts_per_cell > 0,
                             scale_factor / total_counts_per_cell,
                             0.0)
    
    normalized = mc_adata.X * scale_factors[:, np.newaxis]
    
    if hasattr(normalized, 'toarray'):
        normalized = normalized.toarray()
    
    normalized = np.clip(normalized, 0, None)
    normalized_log = np.log1p(normalized)
    
    mc_adata.X = normalized_log
    mc_adata.layers['log_normalized'] = normalized_log.copy()
    
    wgcna_data['metacell_obj'] = mc_adata
    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)
    
    print("NormalizeMetacells complete: metacell expression log-normalized")
    
    return adata
