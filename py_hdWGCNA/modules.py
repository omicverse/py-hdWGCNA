"""
Module eigengenes and connectivity functions for py-hdWGCNA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
from anndata import AnnData


def module_eigengenes(
    adata: AnnData,
    group_by: str = None,
    group_name: str | list = None,
    assay: str = "RNA",
    layer: str = "data",
    use_metacells: bool = False,
    harmonize: bool = True,
    group_by_vars: str | list = None,
    reduction: str = "pca",
    n_pcs: int = 50,
    wgcna_name: str = None,
    n_harmony_runs: int = 1,
    r_harmony_dir: str = None,
):
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)

    if "modules_df" not in wgcna_data:
        raise ValueError("Module assignments not found. Run ConstructNetwork first.")

    modules_df = wgcna_data["modules_df"]

    if use_metacells and "metacell_adata" in wgcna_data:
        mc_adata = wgcna_data["metacell_adata"]
        expr_mat = _get_expr_from_adata(mc_adata, layer)
        genes_all = mc_adata.var_names.tolist()
        cells_use = mc_adata.obs_names.tolist()
        obs_df = mc_adata.obs.copy()
    else:
        expr_mat = _get_expr_from_adata(adata, layer)
        genes_all = adata.var_names.tolist()
        cells_use = adata.obs_names.tolist()
        obs_df = adata.obs.copy()

    network_genes = modules_df["gene_name"].tolist()
    common_genes = [g for g in network_genes if g in genes_all]
    gene_idx_in_expr = [genes_all.index(g) for g in common_genes]
    expr_mat_subset = expr_mat[gene_idx_in_expr, :]

    module_labels_full = np.zeros(len(common_genes), dtype=int)
    for i, gene in enumerate(common_genes):
        mod_row = modules_df[modules_df["gene_name"] == gene]
        if len(mod_row) > 0:
            mod_str = mod_row["module"].values[0]
            if mod_str != "grey":
                try:
                    module_labels_full[i] = int(mod_str.replace("M", ""))
                except ValueError:
                    module_labels_full[i] = 0

    print(
        "Computing module eigengenes (Seurat-compatible ScaleData + SVD PCA + Harmony)..."
    )

    MEs, var_explained, valid_mods, pca_embeddings_dict = (
        _compute_mes_seurat_compatible(
            expr_mat_subset, module_labels_full, exclude_grey=True, n_pcs=n_pcs
        )
    )

    mod_names = [f"M{m}" for m in valid_mods]
    MEs_df = pd.DataFrame(MEs.T, index=cells_use, columns=mod_names)

    if harmonize and group_by_vars is not None:
        if r_harmony_dir is not None and os.path.isdir(r_harmony_dir):
            print(
                f"Harmonizing module eigengenes using R harmony results from: {r_harmony_dir}"
            )
            hME_list = {}
            for idx_m, col in enumerate(mod_names):
                me_pc1 = MEs[idx_m, :]
                r_harmony_file = os.path.join(r_harmony_dir, f"harmony_{col}.csv")
                if os.path.exists(r_harmony_file):
                    r_harm_emb = pd.read_csv(r_harmony_file, index_col=0)
                    r_harm_cells = r_harm_emb.index.tolist()
                    common_hc = [c for c in cells_use if c in r_harm_cells]
                    if len(common_hc) > 0:
                        r_harm_emb = r_harm_emb.loc[common_hc]
                        hme_pc1 = r_harm_emb.iloc[:, 0].values.astype(np.float64)
                        cor_me_hme = np.corrcoef(me_pc1[: len(common_hc)], hme_pc1)[
                            0, 1
                        ]
                        if np.isnan(cor_me_hme) or cor_me_hme < 0:
                            hme_pc1 = -hme_pc1
                        hME_list[col] = hme_pc1
                    else:
                        hME_list[col] = me_pc1
                else:
                    hME_list[col] = me_pc1
            hMEs_df = pd.DataFrame(hME_list, index=cells_use)
        else:
            print(
                "Harmonizing module eigengenes (harmonypy, %d run(s))..."
                % n_harmony_runs
            )
            obs_for_harmony = None
            if isinstance(group_by_vars, str):
                if group_by_vars in obs_df.columns:
                    obs_for_harmony = obs_df.loc[cells_use]
            elif isinstance(group_by_vars, list):
                available = [v for v in group_by_vars if v in obs_df.columns]
                if available:
                    obs_for_harmony = obs_df.loc[cells_use]

            if obs_for_harmony is not None:
                n_harmony_pcs = min(30, n_pcs)

                hME_list = {}
                for idx_m, col in enumerate(mod_names):
                    mod_key = valid_mods[idx_m]
                    pca_emb = pca_embeddings_dict[mod_key]
                    n_available_pcs = pca_emb.shape[1]
                    pca_emb_use = pca_emb[:, : min(n_harmony_pcs, n_available_pcs)]

                    hme_emb = _harmony_correct_single_module(
                        pca_emb_use, obs_for_harmony, group_by_vars
                    )

                    hme_pc1 = hme_emb[:, 0]

                    me_pc1 = MEs[idx_m, :]
                    cor_me_hme = np.corrcoef(me_pc1, hme_pc1)[0, 1]
                    if np.isnan(cor_me_hme) or cor_me_hme < 0:
                        hme_pc1 = -hme_pc1
                    hME_list[col] = hme_pc1

                hMEs_df = pd.DataFrame(hME_list, index=cells_use)
            else:
                hMEs_df = MEs_df.copy()
    else:
        hMEs_df = MEs_df.copy()

    wgcna_data["MEs"] = MEs_df
    wgcna_data["hMEs"] = hMEs_df
    wgcna_data["ME_cells"] = cells_use
    wgcna_data["var_explained_MEs"] = var_explained
    wgcna_data["module_names"] = mod_names

    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)

    print(
        f"ModuleEigengenes complete: {len(mod_names)} module eigengenes for {len(cells_use)} cells"
    )

    return adata


def _compute_mes_seurat_compatible(
    expr_mat: np.ndarray,
    module_labels: np.ndarray,
    exclude_grey: bool = True,
    n_pcs: int = 50,
) -> tuple:
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
    pca_embeddings_dict = {}

    for i, mod in enumerate(unique_modules):
        gene_mask = module_labels == mod
        gene_idx = np.where(gene_mask)[0]

        if len(gene_idx) < 2:
            continue

        mod_expr = expr_mat[gene_idx, :].astype(np.float64)

        n_cells = mod_expr.shape[1]
        n_genes = mod_expr.shape[0]
        mod_centered = mod_expr - np.mean(mod_expr, axis=1, keepdims=True)
        mod_std = np.std(mod_expr, axis=1, ddof=1, keepdims=True)
        mod_std = np.where(mod_std < 1e-10, 1.0, mod_std)
        mod_scaled = mod_centered / mod_std

        clip_val = np.sqrt(n_cells)
        mod_scaled = np.clip(mod_scaled, -clip_val, clip_val)

        nan_mask = np.isnan(mod_scaled)
        if nan_mask.any():
            mod_scaled[nan_mask] = 0.0

        actual_npcs = min(n_pcs, n_genes, n_cells)
        if actual_npcs < 1:
            actual_npcs = 1

        try:
            from scipy.sparse.linalg import svds

            U, s, Vt = svds(mod_scaled.T, k=actual_npcs)
            sort_idx = np.argsort(s)[::-1]
            U = U[:, sort_idx]
            s = s[sort_idx]
            Vt = Vt[sort_idx, :]
            pca_emb = U * s[np.newaxis, :]
        except Exception:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=actual_npcs)
            pca_emb = pca.fit_transform(mod_scaled.T)

        me = pca_emb[:, 0]

        avg_expr = np.mean(mod_expr, axis=0)
        avg_centered = avg_expr - np.mean(avg_expr)
        me_centered = me - np.mean(me)
        denom = np.sqrt(np.sum(avg_centered**2) + 1e-30) * np.sqrt(
            np.sum(me_centered**2) + 1e-30
        )
        pca_cor = np.sum(avg_centered * me_centered) / denom

        if pca_cor < 0:
            me = -me
            pca_emb[:, 0] = -pca_emb[:, 0]

        MEs[i, :] = me
        total_var = np.sum(mod_scaled**2)
        if total_var > 0:
            var_explained[i] = np.sum(me**2) / total_var
        valid_modules.append(mod)
        pca_embeddings_dict[mod] = pca_emb

    MEs = MEs[: len(valid_modules), :]
    var_explained = var_explained[: len(valid_modules)]

    return MEs, var_explained, np.array(valid_modules), pca_embeddings_dict


def module_connectivity(
    adata: AnnData,
    group_by: str = None,
    group_name: str | list = None,
    assay: str = "RNA",
    layer: str = "data",
    use_metacells: bool = False,
    cor_method: str = "bicor",
    sparse: bool = True,
    wgcna_name: str = None,
):
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data, compute_kme

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)

    if "modules_df" not in wgcna_data:
        raise ValueError("Module assignments not found. Run ConstructNetwork first.")

    modules_df = wgcna_data["modules_df"].copy()

    if use_metacells and "metacell_adata" in wgcna_data:
        mc_adata = wgcna_data["metacell_adata"]
        expr_mat = _get_expr_from_adata(mc_adata, layer)
        genes_all = mc_adata.var_names.tolist()
        cells_use = mc_adata.obs_names.tolist()
    else:
        if group_by is not None and group_name is not None:
            if isinstance(group_name, str):
                cell_mask = adata.obs[group_by] == group_name
            else:
                cell_mask = adata.obs[group_by].isin(group_name)
            adata_sub = adata[cell_mask].copy()
            expr_mat = _get_expr_from_adata(adata_sub, layer)
            genes_all = adata_sub.var_names.tolist()
            cells_use = adata_sub.obs_names.tolist()
        else:
            expr_mat = _get_expr_from_adata(adata, layer)
            genes_all = adata.var_names.tolist()
            cells_use = adata.obs_names.tolist()

    if "hMEs" in wgcna_data:
        MEs_df = wgcna_data["hMEs"]
    elif "MEs" in wgcna_data:
        MEs_df = wgcna_data["MEs"]
    else:
        raise ValueError("No MEs found. Run ModuleEigengenes first.")

    me_cells = MEs_df.index.tolist()

    if group_by is not None and group_name is not None and not use_metacells:
        if isinstance(group_name, str):
            cell_mask = adata.obs[group_by] == group_name
        else:
            cell_mask = adata.obs[group_by].isin(group_name)
        subset_cells = adata.obs_names[cell_mask].tolist()
        common_cells = [c for c in subset_cells if c in me_cells]
        MEs_df = MEs_df.loc[common_cells]

    expr_cells = (
        cells_use
        if isinstance(cells_use, list)
        and len(cells_use) > 0
        and not isinstance(cells_use[0], int)
        else None
    )
    if expr_cells is not None:
        common_cells = [c for c in expr_cells if c in MEs_df.index]
        MEs_df = MEs_df.loc[common_cells]
        if len(common_cells) < expr_mat.shape[1]:
            cell_idx = [expr_cells.index(c) for c in common_cells if c in expr_cells]
            expr_mat = expr_mat[:, cell_idx]

    MEs = MEs_df.values.T

    network_genes = modules_df["gene_name"].tolist()
    common_genes = [g for g in network_genes if g in genes_all]

    gene_to_idx = {g: i for i, g in enumerate(genes_all)}
    gene_idx = [gene_to_idx[g] for g in common_genes]
    expr_mat_subset = expr_mat[gene_idx, :]

    print("Computing kME (eigengene-based connectivity)...")
    effective_cor_method = "pearson" if sparse else cor_method
    if sparse and cor_method != "pearson":
        print(
            "  Note: sparse=True forces pearson correlation (matching R corSparse behavior)"
        )
    kME_matrix = compute_kme(expr_mat_subset, MEs, method=effective_cor_method)

    mod_names = wgcna_data.get("module_names", [])
    if len(mod_names) == 0:
        mod_names = [f"M{m}" for m in range(kME_matrix.shape[1])]

    common_set = set(common_genes)
    gene_to_common_idx = {g: i for i, g in enumerate(common_genes)}

    for j, mod_name in enumerate(mod_names):
        col_name = f"kME_{mod_name}"
        kME_col = np.full(len(network_genes), np.nan)
        for i, gene in enumerate(network_genes):
            if gene in common_set:
                kME_col[i] = kME_matrix[gene_to_common_idx[gene], j]
        modules_df[col_name] = kME_col

    wgcna_data["modules_df"] = modules_df
    wgcna_data["kME_computed"] = True
    wgcna_data["kME"] = kME_matrix

    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)

    print(f"ModuleConnectivity complete: kME computed for {len(mod_names)} modules")

    return adata


def reassign_modules(
    adata: AnnData,
    harmonized: bool = True,
    features: list = None,
    new_modules: list = None,
    ignore: bool = False,
    auto_reassign: bool = False,
    wgcna_name: str = None,
):
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)

    modules_df = wgcna_data["modules_df"].copy()

    if harmonized and "hMEs" in wgcna_data:
        MEs = wgcna_data["hMEs"]
    elif "MEs" in wgcna_data:
        MEs = wgcna_data["MEs"]  # noqa: F841
    else:
        raise ValueError("No MEs found. Run ModuleEigengenes first.")

    kME_cols = [c for c in modules_df.columns if c.startswith("kME_")]
    mods = sorted(set(modules_df["module"]))
    mods = [m for m in mods if m != "grey"]
    _genes_use = wgcna_data.get("dat_expr_genes", modules_df["gene_name"].tolist())

    if features is not None and not auto_reassign:
        if new_modules is not None:
            orig_mods = modules_df.set_index("gene_name").loc[features, "module"].values
            if not all(m == "grey" for m in orig_mods) and not ignore:
                non_grey = [m for m in orig_mods if m != "grey"]
                if len(non_grey) > 0:
                    raise ValueError(
                        "Attempting to reassign non-grey genes. Set ignore=True to proceed."
                    )
            reassigned_map = dict(zip(features, new_modules))
            for feat, new_mod in reassigned_map.items():
                idx = modules_df[modules_df["gene_name"] == feat].index
                if len(idx) > 0:
                    modules_df.loc[idx, "module"] = new_mod
                    color_row = modules_df[modules_df["module"] == new_mod].head(1)
                    if len(color_row) > 0:
                        modules_df.loc[idx, "color"] = color_row["color"].values[0]

    elif features is None or auto_reassign:
        neg_indices = []
        for cur_mod in mods:
            col_name = f"kME_{cur_mod}"
            if col_name not in modules_df.columns:
                continue
            cur_mod_df = modules_df[modules_df["module"] == cur_mod].copy()
            neg_mask = cur_mod_df[col_name] < 0
            neg_cur = cur_mod_df[neg_mask]
            if len(neg_cur) > 0:
                neg_indices.extend(neg_cur.index.tolist())

        if len(neg_indices) == 0:
            return adata

        features_to_reassign = modules_df.loc[neg_indices, "gene_name"].tolist()
        kME_vals = modules_df.loc[neg_indices, kME_cols]
        non_grey_kME_cols = [c for c in kME_cols if c != "kME_grey"]

        if len(non_grey_kME_cols) > 0:
            max_kME = kME_vals[non_grey_kME_cols].max(axis=1)
            best_col = kME_vals[non_grey_kME_cols].idxmax(axis=1)
            reassigned = best_col.str.replace("kME_", "")
        else:
            max_kME = pd.Series([np.nan] * len(features_to_reassign))
            reassigned = pd.Series(["grey"] * len(features_to_reassign))

        reassigned[max_kME < 0] = "kME_grey"
        reassigned = reassigned.str.replace("kME_", "")

        reassigned_features = reassigned.index
        for i, feat_idx in enumerate(reassigned_features):
            feat = modules_df.loc[feat_idx, "gene_name"]
            new_mod_val = reassigned.iloc[i]
            row_idx = modules_df[modules_df["gene_name"] == feat].index
            if len(row_idx) > 0:
                modules_df.loc[row_idx, "module"] = new_mod_val
                target_color = modules_df[modules_df["module"] == new_mod_val]["color"]
                if len(target_color) > 0:
                    modules_df.loc[row_idx, "color"] = target_color.values[0]

    wgcna_data["modules_df"] = modules_df
    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)
    print("ReassignModules complete")
    return adata


def reset_module_names(
    adata: AnnData,
    new_name: str = "M",
    reset_levels: bool = False,
    wgcna_name: str = None,
):
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    modules_df = wgcna_data["modules_df"].copy()
    old_mods = sorted(set(modules_df["module"]))

    if "grey" in old_mods:
        nmods = len(old_mods) - 1
    else:
        nmods = len(old_mods)

    new_names = [f"{new_name}{i + 1}" for i in range(nmods)]

    if "grey" in old_mods:
        grey_ind = old_mods.index("grey")
        if grey_ind == 0:
            new_names = ["grey"] + new_names
        elif grey_ind == len(old_mods) - 1:
            new_names = new_names + ["grey"]
        else:
            new_names = new_names[:grey_ind] + ["grey"] + new_names[grey_ind:]

    old_to_new = dict(zip(old_mods, new_names))
    modules_df["module"] = (
        modules_df["module"].map(old_to_new).fillna(modules_df["module"])
    )

    kME_cols_old = [c for c in modules_df.columns if c.startswith("kME_")]
    for old_col in kME_cols_old:
        old_mod_suffix = old_col.replace("kME_", "")
        if old_mod_suffix in old_to_new:
            new_col = f"kME_{old_to_new[old_mod_suffix]}"
            modules_df.rename(columns={old_col: new_col}, inplace=True)

    wgcna_data["modules_df"] = modules_df

    if "MEs" in wgcna_data:
        MEs = wgcna_data["MEs"]
        MEs.columns = [old_to_new.get(c, c) for c in MEs.columns]
        wgcna_data["MEs"] = MEs

    if "hMEs" in wgcna_data:
        hMEs = wgcna_data["hMEs"]
        hMEs.columns = [old_to_new.get(c, c) for c in hMEs.columns]
        wgcna_data["hMEs"] = hMEs

    wgcna_data["module_names"] = [n for n in new_names if n != "grey"]
    adata = set_hdWGCNA_data(adata, wgcna_data, wgcna_name)
    print(f"ResetModuleNames complete: modules renamed to '{new_name}' prefix")
    return adata


def get_modules(adata: AnnData, wgcna_name: str = None) -> pd.DataFrame:
    from .utils import check_wgcna_name, get_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    if "modules_df" not in wgcna_data:
        raise ValueError("Module assignments not found. Run ConstructNetwork first.")
    return wgcna_data["modules_df"]


def get_mes(
    adata: AnnData, harmonized: bool = True, wgcna_name: str = None
) -> pd.DataFrame:
    from .utils import check_wgcna_name, get_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    if harmonized and "hMEs" in wgcna_data:
        return wgcna_data["hMEs"]
    elif "MEs" in wgcna_data:
        return wgcna_data["MEs"]
    else:
        raise ValueError("No MEs found. Run ModuleEigengenes first.")


def get_hub_genes(
    adata: AnnData, n_hubs: int = 10, wgcna_name: str = None
) -> pd.DataFrame:
    from .utils import check_wgcna_name, get_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    modules_df = wgcna_data["modules_df"]
    kME_cols = [
        c for c in modules_df.columns if c.startswith("kME_") and c != "kME_grey"
    ]
    hub_list = []
    for kME_col in kME_cols:
        mod_name = kME_col.replace("kME_", "")
        mod_df = modules_df[["gene_name", "module", kME_col]].copy()
        mod_df = mod_df[mod_df["module"] == mod_name]
        mod_df = mod_df.sort_values(by=kME_col, ascending=False).head(n_hubs)
        mod_df = mod_df.rename(columns={kME_col: "kME"})
        mod_df = mod_df[["gene_name", "module", "kME"]]
        hub_list.append(mod_df)
    if len(hub_list) > 0:
        result = pd.concat(hub_list, ignore_index=True)
    else:
        result = pd.DataFrame(columns=["gene_name", "module", "kME"])
    return result


def get_wgcna_genes(adata: AnnData, wgcna_name: str = None) -> list:
    from .utils import check_wgcna_name, get_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wgcna_data = get_hdWGCNA_data(adata, wgcna_name)
    if "dat_expr_genes" in wgcna_data:
        return wgcna_data["dat_expr_genes"]
    elif "genes_use" in wgcna_data:
        return wgcna_data["genes_use"]
    else:
        raise ValueError("Gene list not found. Run SetupForWGCNA first.")


def _get_expr_from_adata(adata: AnnData, layer: str = "data") -> np.ndarray:
    if layer == "X":
        mat = adata.X
    elif layer == "data":
        mat = adata.raw.X if adata.raw is not None else adata.X
    elif layer in adata.layers:
        mat = adata.layers[layer]
    else:
        mat = adata.X
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    return np.array(mat.T)


def _harmony_correct_module(
    pca_embeddings: np.ndarray,
    obs_df: pd.DataFrame,
    group_by_vars: str | list,
    me_original: np.ndarray,
    n_runs: int = 1,
) -> np.ndarray:
    import harmonypy

    if isinstance(group_by_vars, str):
        group_by_vars = [group_by_vars]

    meta_data = obs_df[group_by_vars].copy()
    meta_data.columns = [str(c) for c in meta_data.columns]

    n_groups = len(meta_data.iloc[:, 0].unique())
    nclust = min(round(pca_embeddings.shape[0] / 30), 100)
    if nclust < n_groups:
        nclust = n_groups

    try:
        ho = harmonypy.run_harmony(
            pca_embeddings,
            meta_data,
            group_by_vars,
            max_iter_harmony=10,
            max_iter_kmeans=20,
            epsilon_harmony=1e-4,
            epsilon_cluster=1e-5,
            nclust=nclust,
            random_state=42,
        )
        return ho.Z_corr
    except Exception:
        me_2d = me_original.reshape(-1, 1) if me_original.ndim == 1 else me_original
        n_cells = me_2d.shape[0]
        design_cols = []
        for gvar in group_by_vars:
            batches = meta_data[gvar].astype("category")
            n_batch = batches.nunique()
            design = np.zeros((n_cells, n_batch))
            codes = batches.cat.codes.values
            for k in range(n_batch):
                design[:, k] = (codes == k).astype(float)
            design_cols.append(design)
        batch_design = (
            np.hstack(design_cols) if len(design_cols) > 1 else design_cols[0]
        )
        from numpy.linalg import lstsq

        coeffs, _, _, _ = lstsq(batch_design, me_2d, rcond=None)
        best_emb = me_2d - batch_design @ coeffs + np.mean(me_2d, axis=0)
        print("  harmonypy failed, used linear regression fallback")
        return best_emb


def _harmony_correct_single_module(
    pca_embeddings: np.ndarray,
    obs_df: pd.DataFrame,
    group_by_vars: str | list,
    max_iter_harmony: int = 10,
    max_iter_kmeans: int = 20,
    epsilon_harmony: float = 1e-4,
    epsilon_cluster: float = 1e-5,
    random_state: int = 42,
) -> np.ndarray:
    try:
        import harmonypy

        if isinstance(group_by_vars, str):
            group_by_vars = [group_by_vars]

        meta_data = obs_df[group_by_vars].copy()
        meta_data.columns = [str(c) for c in meta_data.columns]

        n_groups = len(meta_data.iloc[:, 0].unique())
        nclust = min(round(pca_embeddings.shape[0] / 30), 100)
        if nclust < n_groups:
            nclust = n_groups

        ho = harmonypy.run_harmony(
            pca_embeddings,
            meta_data,
            group_by_vars,
            max_iter_harmony=max_iter_harmony,
            max_iter_kmeans=max_iter_kmeans,
            epsilon_harmony=epsilon_harmony,
            epsilon_cluster=epsilon_cluster,
            nclust=nclust,
            random_state=random_state,
        )

        return ho.Z_corr

    except Exception as e:
        print(f"  harmonypy failed for module: {e}, using uncorrected PCA")
        return pca_embeddings
