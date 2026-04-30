"""
Projection functions for hdWGCNA.

Pure-computation module: project_modules, module_preservation.
Plotting functions are in plotting.py (plot_module_preservation).
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
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


def project_modules(
    adata: AnnData,
    target_adata: AnnData,
    source_hvg: List[str] = None,
    target_hvg: List[str] = None,
    scale_target: bool = True,
    wgcna_name: str = None,
    project_name: str = "projected",
) -> AnnData:
    """
    Project hdWGCNA module eigengenes onto new dataset.

    Replicates R's ProjectModules function.

    Parameters
    ----------
    adata : AnnData
        Source AnnData with hdWGCNA results
    target_adata : AnnData
        Target AnnData to project onto
    source_hvg : list
        Source HVGs (auto-detected if None)
    target_hvg : list
        Target HVGs (auto-detected if None)
    scale_target : bool
        Scale target expression before projection
    wgcna_name : str
        Source experiment name
    project_name : str
        Name for projected experiment in target

    Returns
    -------
    target_adata with projected hMEs stored
    """
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")
    hMEs_source = wd.get("hMEs")

    if modules_df is None:
        raise ValueError("No module assignments found in source.")
    if hMEs_source is None:
        hMEs_source = wd.get("MEs")
    if hMEs_source is None:
        raise ValueError("No module eigengenes found in source.")

    if isinstance(hMEs_source, np.ndarray):
        mod_names = wd.get(
            "module_names", [f"M{i}" for i in range(hMEs_source.shape[1])]
        )
        hMEs_source = pd.DataFrame(hMEs_source, columns=mod_names)

    if source_hvg is None:
        source_hvg = wd.get("dat_expr_genes", [])
        if source_hvg is None or len(source_hvg) == 0:
            source_hvg = [
                g for g in adata.var_names if g in modules_df["gene_name"].values
            ]

    if target_hvg is None:
        target_hvg = list(target_adata.var_names)

    shared_genes = sorted(set(source_hvg) & set(target_hvg))

    if len(shared_genes) < 20:
        raise ValueError(
            f"Only {len(shared_genes)} shared genes between source and target (min 20)."
        )

    print(
        f"Projecting modules onto target ({target_adata.n_obs} cells, {len(shared_genes)} shared genes)"
    )

    source_gene_to_idx = {g: i for i, g in enumerate(shared_genes)}
    target_gene_to_idx = {g: i for i, g in enumerate(shared_genes)}

    src_shared_idx = [source_gene_to_idx[g] for g in shared_genes]
    tgt_shared_idx = [target_gene_to_idx[g] for g in shared_genes]

    src_expr = adata.X[:, src_shared_idx]
    if hasattr(src_expr, "toarray"):
        src_expr = src_expr.toarray()

    tgt_expr = target_adata.X[:, tgt_shared_idx]
    if hasattr(tgt_expr, "toarray"):
        tgt_expr = tgt_expr.toarray()

    if scale_target:
        tgt_mean = tgt_expr.mean(axis=0)
        tgt_std = tgt_expr.std(axis=0) + 1e-8
        tgt_expr = (tgt_expr - tgt_mean) / tgt_std

    src_mean = src_expr.mean(axis=0)
    src_std = src_expr.std(axis=0) + 1e-8
    src_scaled = (src_expr - src_mean) / src_std

    unique_mods = sorted(set(modules_df["module"].unique()) - {"grey"})
    proj_MEs = pd.DataFrame(index=target_adata.obs_names)

    for cur_mod in unique_mods:
        mod_genes = modules_df[modules_df["module"] == cur_mod]["gene_name"].values
        mod_genes_in_src = [g for g in mod_genes if g in source_gene_to_idx]
        mod_genes_shared = [g for g in mod_genes_in_src if g in target_gene_to_idx]

        if len(mod_genes_shared) < 3:
            continue

        src_mod_idx = [source_gene_to_idx[g] for g in mod_genes_in_src]
        tgt_mod_idx = [target_gene_to_idx[g] for g in mod_genes_shared]

        _src_pc1 = np.zeros(src_scaled.shape[0])

        U, S, Vt = np.linalg.svd(src_scaled[:, src_mod_idx], full_matrices=False)
        pc1 = Vt[0, :]
        pc1_norm = np.linalg.norm(pc1)
        if pc1_norm > 0:
            pc1 = pc1 / pc1_norm
        _src_pc1 = src_scaled[:, src_mod_idx] @ pc1  # noqa: F841

        tgt_mod_data = tgt_expr[:, tgt_mod_idx]
        tgt_pc1 = tgt_mod_data @ pc1

        proj_MEs[cur_mod] = tgt_pc1

    grey_mods = [m for m in hMEs_source.columns if m.lower() == "grey"]
    if grey_mods:
        proj_MEs[grey_mods[0]] = 0.0

    if "hdWGCNA" not in target_adata.uns:
        target_adata.uns["hdWGCNA"] = {}

    target_adata.uns["hdWGCNA"][project_name] = {
        "hMEs": proj_MEs,
        "modules_df": modules_df,
        "source_wgcna": wgcna_name,
        "shared_genes": shared_genes,
        "n_shared": len(shared_genes),
    }

    active = target_adata.uns.get("hdWGCNA", {}).get("active_wgcna", None)
    if active is None:
        target_adata.uns["hdWGCNA"]["active_wgcna"] = project_name

    print(f"Projected {len(unique_mods)} modules to target as '{project_name}'.")
    return target_adata


def module_preservation(
    adata_ref: AnnData,
    adata_test: AnnData,
    n_permutations: int = 100,
    random_seed: int = 42,
    wgcna_name: str = None,
) -> pd.DataFrame:
    """
    Assess module preservation via permutation-based Z-summary.

    Replicates R's ModulePreservation function.

    Parameters
    ----------
    adata_ref : AnnData
        Reference dataset with hdWGCNA results
    adata_test : AnnData
        Test dataset
    n_permutations : int
        Number of permutations
    random_seed : int
        Random seed
    wgcna_name : str
        Experiment name

    Returns
    -------
    DataFrame with preservation statistics per module
    """

    wd = _get_wd(adata_ref, wgcna_name)
    modules_df = wd.get("modules_df")

    if modules_df is None:
        raise ValueError("No module data found.")

    TOM = wd.get("TOM")
    if TOM is None:
        raise ValueError("No TOM found in reference. Run construct_network() first.")

    if isinstance(TOM, pd.DataFrame):
        TOM_arr = TOM.values
        tom_genes = list(TOM.index)
    else:
        TOM_arr = np.asarray(TOM)
        tom_genes = wd.get("dat_expr_genes", list(range(TOM_arr.shape[0])))

    ref_gene_set = set(tom_genes)
    test_gene_set = set(adata_test.var_names)
    common_genes = sorted(ref_gene_set & test_gene_set)

    if len(common_genes) < 50:
        raise ValueError(f"Only {len(common_genes)} common genes (min 50).")

    ref_idx = [tom_genes.index(g) for g in common_genes]
    test_idx = [list(adata_test.var_names).index(g) for g in common_genes]

    ref_TOM_sub = TOM_arr[np.ix_(ref_idx, ref_idx)]

    test_X = adata_test.X[:, test_idx]
    if hasattr(test_X, "toarray"):
        test_X = test_X.toarray()

    corr_matrix = np.corrcoef(test_X.T)
    _test_TOM = np.abs(corr_matrix)  # noqa: F841

    np.random.seed(random_seed)

    unique_mods = sorted(set(modules_df["module"].unique()) - {"grey"})

    preservation_stats = []

    for cur_mod in unique_mods:
        mod_row = modules_df[modules_df["module"] == cur_mod]
        mod_genes = (
            mod_row["gene_name"].values.tolist()
            if "gene_name" in mod_row.columns
            else mod_row.index.tolist()
        )
        mod_genes_common = [g for g in mod_genes if g in common_genes]

        if len(mod_genes_common) < 5:
            preservation_stats.append(
                {
                    "module": cur_mod,
                    "Zsummary": np.nan,
                    "medianRank": np.nan,
                    "preservation": "error",
                    "n_genes": len(mod_genes_common),
                }
            )
            continue

        mod_local_idx = [common_genes.index(g) for g in mod_genes_common]
        k = len(mod_local_idx)

        obs_sum_conn = ref_TOM_sub[np.ix_(mod_local_idx, mod_local_idx)].sum() - k
        obs_median_rank = np.median(
            [np.median(ref_TOM_sub[i, mod_local_idx]) for i in mod_local_idx]
        )

        perm_sums = np.zeros(n_permutations)
        perm_med_ranks = np.zeros(n_permutations)

        rng = np.random.default_rng(random_seed)
        for perm_i in range(n_permutations):
            perm_indices = rng.choice(len(common_genes), size=k, replace=False)
            perm_TOM = ref_TOM_sub[np.ix_(perm_indices, perm_indices)]
            perm_sums[perm_i] = perm_TOM.sum() - k
            perm_med_ranks[perm_i] = np.median(
                [np.median(ref_TOM_sub[p, perm_indices]) for p in perm_indices]
            )

        perm_mean = perm_sums.mean()
        perm_std = perm_sums.std()
        Zsummary = (obs_sum_conn - perm_mean) / perm_std if perm_std > 0 else 0

        perm_mr_mean = perm_med_ranks.mean()
        perm_mr_std = perm_med_ranks.std()
        __medianRank_Z = (
            (obs_median_rank - perm_mr_mean) / perm_mr_std if perm_mr_std > 0 else 0
        )  # noqa: F841

        if Zsummary >= 10:
            preserv_cat = "highly preserved"
        elif Zsummary >= 5:
            preserv_cat = "moderately preserved"
        elif Zsummary >= 2:
            preserv_cat = "weakly preserved"
        else:
            preserv_cat = "non-preserved"

        preservation_stats.append(
            {
                "module": cur_mod,
                "Zsummary": round(Zsummary, 4),
                "medianRank": round(obs_median_rank, 4),
                "preservation": preserv_cat,
                "n_genes": len(mod_genes_common),
            }
        )

    preserv_df = pd.DataFrame(preservation_stats)
    print("\nModule Preservation Summary:")
    print(preserv_df.to_string(index=False))
    return preserv_df
