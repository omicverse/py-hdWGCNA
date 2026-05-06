"""
Analysis functions for hdWGCNA.

Pure-computation module: DME (Differential Module Expression) analysis,
module-trait correlation. Plotting functions are in plotting.py.
"""

from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats
from anndata import AnnData


def _get_wgcna_name(adata: AnnData, wgcna_name: str = None) -> str:
    if wgcna_name is None:
        wgcna_name = adata.uns.get("hdWGCNA", {}).get("active_wgcna", None)
    if wgcna_name is None:
        raise ValueError("No active hdWGCNA experiment found.")
    return wgcna_name


def _get_wd(adata: AnnData, wgcna_name: str = None) -> dict:
    wn = _get_wgcna_name(adata, wgcna_name)
    return adata.uns["hdWGCNA"][wn]


def find_dmes(
    adata: AnnData,
    group_by: str,
    group1: str,
    group2: str,
    features: str = "hMEs",
    test: str = "wilcox",
    logfc_threshold: float = 0.25,
    min_pct: float = 0.1,
    wgcna_name: str = None,
) -> pd.DataFrame:
    """
    Find Differential Module Expression between two groups.

    Replicates R's FindDMEs function.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results and group column in obs
    group_by : str
        Column name for grouping
    group1 : str
        Reference group value
    group2 : str
        Comparison group value
    features : str
        'hMEs', 'MEs', 'scores', or 'average'
    test : str
        'wilcox' or 'ttest'
    logfc_threshold : float
        Minimum log2FC threshold
    min_pct : float
        Minimum fraction of cells expressing the feature
    wgcna_name : str
        Experiment name

    Returns
    -------
    DataFrame with DME statistics per module
    """
    wd = _get_wd(adata, wgcna_name)

    me_key_map = {
        "hMEs": "hMEs",
        "MEs": "MEs",
        "scores": "module_scores",
        "average": "avg_module_expr",
    }
    me_key = me_key_map.get(features, "hMEs")
    MEs = wd.get(me_key)

    if MEs is None:
        raise ValueError(f"No {features} found. Run module_eigengenes() first.")

    if isinstance(MEs, np.ndarray):
        mod_names = wd.get("module_names", [f"M{i}" for i in range(MEs.shape[1])])
        MEs = pd.DataFrame(MEs, columns=mod_names)

    if group_by not in adata.obs.columns:
        raise ValueError(f"Grouping column '{group_by}' not found in adata.obs.")

    mask1 = adata.obs[group_by] == group1
    mask2 = adata.obs[group_by] == group2

    n1 = int(mask1.sum())
    n2 = int(mask2.sum())

    if n1 < 3 or n2 < 3:
        raise ValueError(
            f"Insufficient cells: {group1}={n1}, {group2}={n2} (min 3 each)"
        )

    MEs_1 = MEs.loc[mask1]
    MEs_2 = MEs.loc[mask2]

    modules_df = wd.get("modules_df")
    if modules_df is not None and isinstance(modules_df, pd.DataFrame):
        mod_colors_dict = dict(zip(modules_df["module"], modules_df["color"]))
    else:
        mod_colors_dict = {}

    results = []
    for cur_mod in MEs.columns:
        v1 = MEs_1[cur_mod].dropna().values
        v2 = MEs_2[cur_mod].dropna().values

        pct1 = np.mean(v1 != 0) if len(v1) > 0 else 0
        pct2 = np.mean(v2 != 0) if len(v2) > 0 else 0
        avg_pct = (pct1 + pct2) / 2  # noqa: F841

        mean1 = np.mean(v1) if len(v1) > 0 else 0
        mean2 = np.mean(v2) if len(v2) > 0 else 0
        log2fc = mean2 - mean1

        if len(v1) > 1 and len(v2) > 1:
            try:
                if test.lower() == "wilcox":
                    stat_val, p_val = stats.mannwhitneyu(
                        v1, v2, alternative="two-sided"
                    )
                elif test.lower() == "ttest":
                    stat_val, p_val = stats.ttest_ind(v1, v2, equal_var=False)
                else:
                    p_val = 1.0
                    _stat_val = 0
            except Exception:
                p_val = 1.0
                _stat_val = 0
        else:
            p_val = 1.0
            _stat_val = 0

        from statsmodels.stats.multitest import multipletests

        results.append(
            {
                "module": cur_mod,
                "p_val": p_val,
                "avg_log2FC": log2fc,
                "pct.1": round(pct1, 4),
                "pct.2": round(pct2, 4),
                "p_val_adj": 1.0,
                "color": mod_colors_dict.get(cur_mod, "#0072B2"),
                f"n_{group1}": n1,
                f"n_{group2}": n2,
            }
        )

    dme_df = pd.DataFrame(results)

    if len(dme_df) > 0:
        _, dme_df["p_val_adj"], _, _ = multipletests(
            dme_df["p_val"].fillna(1).values, method="fdr_bh"
        )

    return dme_df


def find_all_dmes(
    adata: AnnData,
    group_by: str,
    features: str = "hMEs",
    test: str = "wilcox",
    logfc_threshold: float = 0.25,
    min_pct: float = 0.1,
    wgcna_name: str = None,
) -> Dict[str, pd.DataFrame]:
    """
    Find DMEs for all pairwise comparisons within a grouping variable.

    Replicates R's FindAllDMEs function.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    group_by : str
        Column name for grouping
    features : str
        Which MEs to use
    test : str
        Statistical test ('wilcox' or 'ttest')
    logfc_threshold : float
        Minimum log2FC threshold
    min_pct : float
        Minimum expression percentage
    wgcna_name : str
        Experiment name

    Returns
    -------
    Dict mapping comparison string to DME DataFrame
    """
    groups = sorted(adata.obs[group_by].unique().tolist())

    all_results = {}
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1 :]:
            comp_str = f"{g1}_vs_{g2}"
            print(f"  Computing DME: {comp_str}")
            try:
                dme_res = find_dmes(
                    adata,
                    group_by=group_by,
                    group1=g1,
                    group2=g2,
                    features=features,
                    test=test,
                    logfc_threshold=logfc_threshold,
                    min_pct=min_pct,
                    wgcna_name=wgcna_name,
                )
                dme_res["comparison"] = comp_str
                dme_res["group_by"] = group_by
                all_results[comp_str] = dme_res
            except Exception as e:
                print(f"  WARNING: Failed to compute DME for {comp_str}: {e}")

    return all_results


def module_trait_correlation(
    adata: AnnData,
    trait_cols: List[str] = None,
    features: str = "hMEs",
    method: str = "pearson",
    wgcna_name: str = None,
) -> Optional[Dict]:
    """
    Compute correlation between module eigengenes and trait variables.

    Replicates R's ModuleTraitCorrelation function.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results and trait columns in obs
    trait_cols : list
        Specific trait columns (default: all numeric obs columns)
    features : str
        'hMEs', 'MEs', 'scores', or 'average'
    method : str
        Correlation method ('pearson' or 'spearman')
    wgcna_name : str
        Experiment name

    Returns
    -------
    Dict with keys: cor, pval, fdr (each mapping trait -> DataFrame)
    """
    wd = _get_wd(adata, wgcna_name)

    me_key_map = {
        "hMEs": "hMEs",
        "MEs": "MEs",
        "scores": "module_scores",
        "average": "avg_module_expr",
    }
    me_key = me_key_map.get(features, "hMEs")
    MEs = wd.get(me_key)

    if MEs is None:
        raise ValueError(f"No {features} found. Run module_eigengenes() first.")

    if isinstance(MEs, np.ndarray):
        mod_names = wd.get("module_names", [f"M{i}" for i in range(MEs.shape[1])])
        MEs = pd.DataFrame(MEs, columns=mod_names)

    if trait_cols is None or len(trait_cols) == 0:
        trait_cols = [
            c
            for c in adata.obs.columns
            if c not in ["cell_type", "seurat_clusters", "orig.ident"]
            and pd.api.types.is_numeric_dtype(adata.obs[c])
        ]

    if len(trait_cols) == 0:
        print("No numeric trait columns found.")
        return None

    cor_results = {}
    pval_results = {}
    fdr_results = {}

    for tcol in trait_cols:
        if tcol not in adata.obs.columns:
            continue
        trait_vals = adata.obs[tcol].astype(float).values

        valid_mask = ~np.isnan(trait_vals) & ~pd.isnull(trait_vals)
        if valid_mask.sum() < 10:
            continue

        trait_clean = trait_vals[valid_mask]
        MEs_clean = MEs.loc[valid_mask].copy()

        cor_series = []
        pval_series = []
        for mod in MEs_clean.columns:
            me_vals = MEs_clean[mod].values
            me_valid = ~np.isnan(me_vals)
            t_valid = ~np.isnan(trait_clean)
            both_valid = me_valid & t_valid

            if both_valid.sum() < 5:
                cor_series.append(np.nan)
                pval_series.append(np.nan)
                continue

            r, p = (
                stats.pearsonr(me_vals[both_valid], trait_clean[both_valid])
                if method == "pearson"
                else stats.spearmanr(me_vals[both_valid], trait_clean[both_valid])
            )
            cor_series.append(r)
            pval_series.append(p)

        cor_df = pd.DataFrame({tcol: cor_series}, index=MEs_clean.columns)
        cor_df.index.name = "Module"

        pval_df = pd.DataFrame({tcol: pval_series}, index=MEs_clean.columns)
        pval_df.index.name = "Module"

        flat_pvals = pval_df[tcol].dropna().values
        if len(flat_pvals) > 0:
            from statsmodels.stats.multitest import multipletests

            _, fdr_vals, _, _ = multipletests(flat_pvals, method="fdr_bh")

            fdr_df = pd.DataFrame({tcol: np.nan}, index=pval_df.index)
            valid_idx = pval_df[tcol].notna()
            fdf_valid = fdr_df.loc[valid_idx]
            fdf_valid[tcol] = fdr_vals
            fdr_df.loc[valid_idx] = fdf_valid
        else:
            fdr_df = pd.DataFrame({tcol: np.nan}, index=pval_df.index)

        cor_results[tcol] = cor_df
        pval_results[tcol] = pval_df
        fdr_results[tcol] = fdr_df

    if len(cor_results) == 0:
        return None

    return {"cor": cor_results, "pval": pval_results, "fdr": fdr_results}


def overlap_modules_degs(
    adata: AnnData,
    deg_df: pd.DataFrame,
    fc_cutoff: float = 0.5,
    group_col: str = "cluster",
    wgcna_name: str = None,
) -> pd.DataFrame:
    """
    Perform Fisher's Exact Test for overlap between DEGs and hdWGCNA modules.

    Replicates R's OverlapModulesDEGs function.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    deg_df : DataFrame
        DEG table with columns: gene, p_val_adj, avg_log2FC, and a group column
    fc_cutoff : float
        Log fold change cutoff for DEGs
    group_col : str
        Column name in deg_df containing group info
    wgcna_name : str
        Experiment name

    Returns
    -------
    DataFrame with overlap statistics per module-group pair
    """
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")
    if modules_df is None:
        raise ValueError("No module data found.")

    deg_df = deg_df.copy()
    deg_df["group"] = deg_df[group_col]
    cell_groups = deg_df["group"].unique()

    mods = sorted(modules_df["module"].unique())
    mods = [m for m in mods if m != "grey"]

    if fc_cutoff >= 0:
        deg_df = deg_df[deg_df["avg_log2FC"] >= fc_cutoff]
    else:
        deg_df = deg_df[deg_df["avg_log2FC"] <= fc_cutoff]
        deg_df["avg_log2FC"] = -1 * deg_df["avg_log2FC"]
        fc_cutoff = -1 * fc_cutoff

    genome_size = adata.n_vars

    results = []
    for cur_mod in mods:
        cur_module_genes = set(
            modules_df[modules_df["module"] == cur_mod]["gene_name"].tolist()
        )
        for cur_group in cell_groups:
            cur_degs = deg_df[
                (deg_df["group"] == cur_group)
                & (deg_df["p_val_adj"] <= 0.05)
                & (deg_df["avg_log2FC"] > fc_cutoff)
            ]["gene"].tolist()
            cur_degs = set(str(g) for g in cur_degs)

            intersection = cur_module_genes & cur_degs
            n_intersect = len(intersection)

            n_mod_only = len(cur_module_genes) - n_intersect
            n_deg_only = len(cur_degs) - n_intersect
            n_neither = genome_size - n_intersect - n_mod_only - n_deg_only

            if n_neither < 0:
                n_neither = 0

            try:
                _, p_val = stats.fisher_exact(
                    [[n_intersect, n_mod_only], [n_deg_only, n_neither]],
                    alternative="greater",
                )
            except Exception:
                p_val = 1.0

            a_val = len(cur_module_genes)
            b_val = genome_size - len(cur_module_genes)
            c_val = len(cur_degs)
            d_val = genome_size - len(cur_degs)
            if a_val > 0 and c_val > 0 and b_val > 0 and d_val > 0:
                odds_ratio = (a_val * d_val) / (b_val * c_val)
            else:
                odds_ratio = 0.0

            if len(cur_module_genes) > 0 and len(cur_degs) > 0:
                jaccard = n_intersect / len(cur_module_genes | cur_degs)
            else:
                jaccard = 0.0

            mod_color = (
                modules_df[modules_df["module"] == cur_mod]["color"].iloc[0]
                if len(modules_df[modules_df["module"] == cur_mod]) > 0
                else "#0072B2"
            )

            results.append(
                {
                    "module": cur_mod,
                    "group": cur_group,
                    "color": mod_color,
                    "odds_ratio": odds_ratio,
                    "pval": p_val,
                    "Jaccard": jaccard,
                    "size_intersection": n_intersect,
                }
            )

    overlap_df = pd.DataFrame(results)
    if len(overlap_df) > 0:
        from statsmodels.stats.multitest import multipletests

        _, overlap_df["fdr"], _, _ = multipletests(
            overlap_df["pval"].fillna(1).values, method="fdr_bh"
        )

        def _stars(p):
            if pd.isna(p):
                return ""
            if p < 0.001:
                return "***"
            if p < 0.01:
                return "**"
            if p < 0.05:
                return "*"
            return ""

        overlap_df["Significance"] = overlap_df["fdr"].apply(_stars)
        overlap_df["module"] = pd.Categorical(
            overlap_df["module"], categories=mods, ordered=True
        )
        overlap_df = overlap_df[
            [
                "module",
                "group",
                "color",
                "odds_ratio",
                "pval",
                "fdr",
                "Significance",
                "Jaccard",
                "size_intersection",
            ]
        ]

    return overlap_df


def module_expr_score(
    adata: AnnData,
    n_genes: int = 25,
    wgcna_name: str = None,
) -> AnnData:
    """
    Compute module expression scores using Seurat AddModuleScore approach.

    Replicates R's ModuleExprScore function.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    n_genes : int or str
        Number of top hub genes per module. Use 'all' for all genes.
    wgcna_name : str
        Experiment name

    Returns
    -------
    AnnData with module scores stored in hdWGCNA data
    """
    wd = _get_wd(adata, wgcna_name)
    wgcna_name = _get_wgcna_name(adata, wgcna_name)
    modules_df = wd.get("modules_df")
    if modules_df is None:
        raise ValueError("No module data found.")

    mods = sorted(modules_df["module"].unique())
    mods = [m for m in mods if m != "grey"]

    gene_lists = {}
    for cur_mod in mods:
        cur_mod_df = modules_df[modules_df["module"] == cur_mod]
        if n_genes == "all":
            gene_lists[cur_mod] = cur_mod_df["gene_name"].tolist()
        else:
            kme_col = f"kME_{cur_mod}"
            if kme_col in cur_mod_df.columns:
                cur_mod_df = cur_mod_df.nlargest(n_genes, kme_col)
            gene_lists[cur_mod] = cur_mod_df["gene_name"].tolist()

    from scipy.sparse import issparse

    if issparse(adata.X):
        expr = adata.X.toarray()
    else:
        expr = adata.X.copy()

    gene_names = list(adata.var_names)
    all_control_genes = list(range(expr.shape[1]))

    mod_scores = {}
    for cur_mod in mods:
        cur_genes = gene_lists[cur_mod]
        feature_indices = [gene_names.index(g) for g in cur_genes if g in gene_names]
        if len(feature_indices) == 0:
            continue

        n_bins = 25
        gene_means = np.array(expr.mean(axis=0)).flatten()
        bin_edges = np.quantile(
            gene_means[gene_means > 0], np.linspace(0, 1, n_bins + 1)
        )
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        control_indices = []
        for fi in feature_indices:
            gmean = gene_means[fi]
            bin_idx = np.searchsorted(bin_edges, gmean) - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))
            candidates = [
                i
                for i in all_control_genes
                if i not in feature_indices
                and bin_edges[bin_idx] <= gene_means[i] < bin_edges[bin_idx + 1]
            ]
            if len(candidates) > 0:
                n_ctrl = min(len(candidates), len(feature_indices))
                rng = np.random.RandomState(42)
                ctrl_idx = rng.choice(candidates, size=n_ctrl, replace=False).tolist()
                control_indices.extend(ctrl_idx)

        control_indices = list(set(control_indices))
        if len(control_indices) == 0:
            continue

        feature_expr = expr[:, feature_indices].mean(axis=1)
        control_expr = expr[:, control_indices].mean(axis=1)
        mod_scores[cur_mod] = feature_expr - control_expr

    if len(mod_scores) > 0:
        scores_df = pd.DataFrame(mod_scores, index=adata.obs_names)
        col_order = [m for m in mods if m in scores_df.columns]
        scores_df = scores_df[col_order]
        adata.uns["hdWGCNA"][wgcna_name]["module_scores"] = scores_df

    return adata


def avg_module_expr(
    adata: AnnData,
    n_genes: int = 25,
    wgcna_name: str = None,
) -> AnnData:
    """
    Compute average expression for each co-expression module.

    Replicates R's AvgModuleExpr function.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    n_genes : int or str
        Number of top hub genes per module. Use 'all' for all genes.
    wgcna_name : str
        Experiment name

    Returns
    -------
    AnnData with average module expression stored in hdWGCNA data
    """
    wd = _get_wd(adata, wgcna_name)
    wgcna_name = _get_wgcna_name(adata, wgcna_name)
    modules_df = wd.get("modules_df")
    if modules_df is None:
        raise ValueError("No module data found.")

    mods = sorted(modules_df["module"].unique())
    mods = [m for m in mods if m != "grey"]

    gene_lists = {}
    for cur_mod in mods:
        cur_mod_df = modules_df[modules_df["module"] == cur_mod]
        if n_genes == "all":
            gene_lists[cur_mod] = cur_mod_df["gene_name"].tolist()
        else:
            kme_col = f"kME_{cur_mod}"
            if kme_col in cur_mod_df.columns:
                cur_mod_df = cur_mod_df.nlargest(n_genes, kme_col)
            gene_lists[cur_mod] = cur_mod_df["gene_name"].tolist()

    from scipy.sparse import issparse

    if issparse(adata.X):
        expr = adata.X.toarray()
    else:
        expr = adata.X.copy()

    gene_names = list(adata.var_names)

    avg_dict = {}
    for cur_mod in mods:
        cur_genes = gene_lists[cur_mod]
        feature_indices = [gene_names.index(g) for g in cur_genes if g in gene_names]
        if len(feature_indices) == 0:
            continue
        avg_dict[cur_mod] = expr[:, feature_indices].mean(axis=1)

    if len(avg_dict) > 0:
        avg_df = pd.DataFrame(avg_dict, index=adata.obs_names)
        col_order = [m for m in mods if m in avg_df.columns]
        avg_df = avg_df[col_order]
        adata.uns["hdWGCNA"][wgcna_name]["avg_module_expr"] = avg_df

    return adata
