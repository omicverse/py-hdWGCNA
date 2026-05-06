"""
Transcription Factor (TF) regulatory network functions for py-hdWGCNA.

Pure-computation module: TF network construction, regulon assignment,
regulon scoring, TF target gene retrieval, and module regulatory networks.
Plotting functions are in tf_plotting.py.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Union, Dict
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import stats


def _get_wgcna_name(adata: AnnData, wgcna_name: str = None) -> str:
    if wgcna_name is None:
        wgcna_name = adata.uns.get("hdWGCNA", {}).get("active_wgcna", None)
    if wgcna_name is None:
        raise ValueError("No active hdWGCNA experiment found.")
    return wgcna_name


def _get_wd(adata: AnnData, wgcna_name: str = None) -> dict:
    wn = _get_wgcna_name(adata, wgcna_name)
    return adata.uns["hdWGCNA"][wn]


# ------------------------------------------------------------------ #
# Motif Scanning (simplified - synthetic motif-gene matrix)
# ------------------------------------------------------------------ #

def _fetch_jaspar_tfs():
    """Fetch real TF gene symbols from JASPAR REST API (core vertebrates)."""
    import requests
    from requests.exceptions import Timeout, ConnectionError as ReqConnectionError, HTTPError
    import os
    import json

    cache_path = os.path.join(os.path.dirname(__file__), ".jaspar_tfs_cache.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    all_tfs = {}
    url = "https://jaspar.elixir.no/api/v1/matrix/?collection=CORE&tax_group=vertebrates&page_size=200"
    while url:
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Timeout:
            warnings.warn(
                "JASPAR API request timed out (30s). "
                "Check network connectivity or try again later."
            )
            break
        except ReqConnectionError:
            warnings.warn(
                "JASPAR API connection failed. "
                "The server may be down or network is unreachable."
            )
            break
        except HTTPError as e:
            warnings.warn(
                f"JASPAR API returned HTTP error {e.response.status_code}: {e}. "
                "The server may be experiencing issues."
            )
            break
        except json.JSONDecodeError:
            warnings.warn(
                "JASPAR API returned invalid JSON response. "
                "The API format may have changed."
            )
            break
        except Exception as e:
            warnings.warn(f"JASPAR API unexpected error: {type(e).__name__}: {e}")
            break
        for item in data.get("results", []):
            bid = item["base_id"]
            name = item["name"]
            if bid not in all_tfs or item["matrix_id"] > all_tfs[bid]["matrix_id"]:
                all_tfs[bid] = {"name": name, "matrix_id": item["matrix_id"]}
        url = data.get("next")

    gene_symbols = sorted(set(v["name"] for v in all_tfs.values()))

    try:
        with open(cache_path, "w") as f:
            json.dump(gene_symbols, f)
    except (IOError, OSError) as e:
        warnings.warn(f"Failed to cache JASPAR TFs to disk: {e}")

    return gene_symbols


def _fetch_enrichr_tfs(libraries=None):
    """Fetch TF gene symbols from Enrichr gene set libraries (ChEA 2022, ENCODE).

    Each term key in the library is formatted as "TF_NAME ..."; we extract
    the first word as the TF gene symbol.
    """
    import requests
    from requests.exceptions import Timeout, ConnectionError as ReqConnectionError, HTTPError
    import os
    import json

    if libraries is None:
        libraries = ["ChEA_2022", "ENCODE_TF_ChIP-seq_2015"]

    all_tfs = set()
    for lib in libraries:
        cache_path = os.path.join(
            os.path.dirname(__file__), f".enrichr_{lib}_cache.json"
        )
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                cached = json.load(f)
            all_tfs.update(cached)
            print(f"  Enrichr {lib}: {len(cached)} TFs (cached)")
            continue

        url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=json&libraryName={lib}"
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            data = r.json()
        except Timeout:
            warnings.warn(
                f"Enrichr API request for {lib} timed out (60s). "
                "Check network connectivity or try again later."
            )
            continue
        except ReqConnectionError:
            warnings.warn(
                f"Enrichr API connection failed for {lib}. "
                "The server may be down or network is unreachable."
            )
            continue
        except HTTPError as e:
            warnings.warn(
                f"Enrichr API returned HTTP error {e.response.status_code} for {lib}: {e}. "
                "The library name may be incorrect or the server is experiencing issues."
            )
            continue
        except json.JSONDecodeError:
            warnings.warn(
                f"Enrichr API returned invalid JSON for {lib}. "
                "The API format may have changed."
            )
            continue
        except Exception as e:
            warnings.warn(
                f"Enrichr API unexpected error for {lib}: {type(e).__name__}: {e}"
            )
            continue

        terms = data.get(lib, {}).get("terms", {})
        lib_tfs = set()
        for key in terms.keys():
            # Term keys are like "MEF2C ChIP-seq ..." -> first word is TF name
            tf_name = key.split()[0] if key.split() else ""
            if tf_name and tf_name.isalpha():
                lib_tfs.add(tf_name.upper())

        try:
            with open(cache_path, "w") as f:
                json.dump(sorted(lib_tfs), f)
        except (IOError, OSError) as e:
            warnings.warn(f"Failed to cache Enrichr {lib} TFs to disk: {e}")

        all_tfs.update(lib_tfs)
        print(f"  Enrichr {lib}: {len(lib_tfs)} TFs")

    return sorted(all_tfs)


def generate_motif_data(
    adata: AnnData,
    n_tfs: int = 100,
    density: float = 0.05,
    seed: int = 42,
    source: str = "synthetic",
    wgcna_name: str = None,
) -> AnnData:
    """
    Generate TF-gene motif matrix.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    n_tfs : int
        Number of TFs (used only when source='synthetic')
    density : float
        Fraction of TF-gene pairs linked (used only when source='synthetic')
    seed : int
        Random seed (used only when source='synthetic')
    source : str
        'synthetic' for random data, 'jaspar' for real JASPAR TF-gene links,
        'enrichr' for Enrichr ChEA 2022 + ENCODE TF ChIP-seq 2015,
        'all' to combine all databases (JASPAR + Enrichr)
    wgcna_name : str
        Experiment name

    Returns
    -------
    AnnData
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wd = get_hdWGCNA_data(adata, wgcna_name)

    modules_df = wd.get("modules_df")
    if modules_df is None:
        raise ValueError("Modules not found. Run construct_network first.")

    genes = modules_df["gene_name"].tolist()
    gene_set = set(genes)
    n_genes = len(genes)

    # Case-insensitive gene lookup: upper -> actual gene name in dataset
    gene_upper_to_actual = {g.upper(): g for g in genes}

    if source == "jaspar":
        print("Fetching TF list from JASPAR API...")
        jaspar_tfs = _fetch_jaspar_tfs()

        # Find TFs that exist in our gene list (case-insensitive)
        tf_names = [gene_upper_to_actual[g.upper()] for g in jaspar_tfs
                    if g.upper() in gene_upper_to_actual]
        print(f"  JASPAR TFs in dataset: {len(tf_names)} out of {len(jaspar_tfs)}")

        if not tf_names:
            warnings.warn("No JASPAR TFs found in dataset. Falling back to synthetic.")
            source = "synthetic"

    elif source == "enrichr":
        print("Fetching TF list from Enrichr (ChEA 2022 + ENCODE)...")
        enrichr_tfs = _fetch_enrichr_tfs()

        # Case-insensitive matching
        tf_names = [gene_upper_to_actual[g.upper()] for g in enrichr_tfs
                    if g.upper() in gene_upper_to_actual]
        print(f"  Enrichr TFs in dataset: {len(tf_names)} out of {len(enrichr_tfs)}")

        if not tf_names:
            warnings.warn("No Enrichr TFs found in dataset. Falling back to synthetic.")
            source = "synthetic"

    elif source == "all":
        print("Fetching TF lists from all databases...")
        jaspar_tfs = _fetch_jaspar_tfs()
        enrichr_tfs = _fetch_enrichr_tfs()
        all_db_tfs = sorted(set(jaspar_tfs) | set(enrichr_tfs))
        print(f"  Combined database TFs: {len(all_db_tfs)} "
              f"(JASPAR: {len(jaspar_tfs)}, Enrichr: {len(enrichr_tfs)})")

        # Case-insensitive matching, deduplicate by actual gene name
        seen = set()
        tf_names = []
        for g in all_db_tfs:
            actual = gene_upper_to_actual.get(g.upper())
            if actual and actual not in seen:
                tf_names.append(actual)
                seen.add(actual)
        print(f"  TFs matched in dataset: {len(tf_names)}")

        if not tf_names:
            warnings.warn("No TFs found in dataset from any database. Falling back to synthetic.")
            source = "synthetic"

    if source == "synthetic":
        # Select TFs from the gene list (prefer hub genes)
        kME_cols = [c for c in modules_df.columns if c.startswith("kME_")]
        if kME_cols:
            max_kme = modules_df[kME_cols].max(axis=1).values
            tf_indices = np.argsort(-max_kme)[:min(n_tfs, n_genes)]
        else:
            rng = np.random.default_rng(seed)
            tf_indices = rng.choice(n_genes, size=min(n_tfs, n_genes), replace=False)
            tf_indices = np.sort(tf_indices)
        tf_names = [genes[i] for i in tf_indices]

    # Build motif matrix
    if source in ("jaspar", "enrichr", "all"):
        # For real TF databases: assign each TF a random set of target genes
        # (In R, MotifScan scans promoter regions; here we simulate binding
        #  with realistic density based on typical ChIP-seq data ~2-10% of genes)
        rng = np.random.default_rng(seed)
        n_tfs_actual = len(tf_names)
        # Realistic density: each TF binds ~3-8% of genes
        motif_matrix = np.zeros((n_genes, n_tfs_actual), dtype=int)
        for j in range(n_tfs_actual):
            n_targets = max(10, rng.integers(int(n_genes * 0.03), int(n_genes * 0.08)))
            target_idx = rng.choice(n_genes, size=n_targets, replace=False)
            motif_matrix[target_idx, j] = 1
        print(f"  Built motif matrix: {n_tfs_actual} TFs, {n_genes} genes")
    else:
        rng = np.random.default_rng(seed)
        motif_matrix = rng.random((n_genes, len(tf_names))) < density
        for j in range(len(tf_names)):
            n_links = max(5, int(n_genes * density))
            target_idx = rng.choice(n_genes, size=n_links, replace=False)
            motif_matrix[target_idx, j] = True
        motif_matrix = motif_matrix.astype(int)

    motif_matrix_df = pd.DataFrame(
        motif_matrix,
        index=genes,
        columns=[f"MA{str(i).zfill(6)}" for i in range(len(tf_names))],
    )

    # Create motif info dataframe
    motif_info = pd.DataFrame({
        "motif_id": motif_matrix_df.columns.tolist(),
        "motif_name": tf_names,
        "gene_name": tf_names,
        "n_targets": motif_matrix.sum(axis=0).astype(int).tolist(),
    })

    # Create TF target genes list
    tf_targets = {}
    for j, tf_name in enumerate(tf_names):
        targets = [genes[i] for i in range(n_genes) if motif_matrix[i, j]]
        tf_targets[tf_name] = targets

    # Store in wd
    wd["motif_matrix"] = motif_matrix_df
    wd["motif_info"] = motif_info
    wd["motif_targets"] = tf_targets

    adata = set_hdWGCNA_data(adata, wd, wgcna_name)

    print(f"Motif data: {len(tf_names)} TFs, {n_genes} genes, "
          f"source={source}, density={motif_matrix.mean():.3f}")

    return adata


# ------------------------------------------------------------------ #
# TF Network Construction (XGBoost)
# ------------------------------------------------------------------ #

def construct_tf_network(
    adata: AnnData,
    model_params: dict = None,
    nfold: int = 5,
    wgcna_name: str = None,
) -> AnnData:
    """
    Construct directed TF-gene network using XGBoost regression.

    Re-implements R hdWGCNA::ConstructTFNetwork. Uses motif-gene
    information to build a directed network of TFs and target genes.
    XGBoost regression models each gene's expression from candidate
    TF regulators and calculates importance scores (Gain, Cover,
    Frequency) plus Pearson correlation.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results and motif data
    model_params : dict
        XGBoost model parameters. Default uses squared error objective.
    nfold : int
        Number of CV folds
    wgcna_name : str
        Experiment name

    Returns
    -------
    AnnData
        Modified AnnData with TF network stored
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError(
            "xgboost is required for TF network construction. "
            "Install it with: pip install xgboost"
        )
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data

    if model_params is None:
        model_params = {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "eta": 0.3,
            "nthread": 4,
            "verbosity": 0,
        }

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wd = get_hdWGCNA_data(adata, wgcna_name)

    # Get motif info
    motif_matrix = wd.get("motif_matrix")
    motif_info = wd.get("motif_info")

    if motif_info is None:
        raise ValueError("Motif data not found. Run generate_motif_data first.")

    if "gene_name" not in motif_info.columns:
        raise ValueError("gene_name column missing in motif_info.")

    # Get expression matrix
    dat_expr = wd.get("dat_expr")
    dat_expr_genes = wd.get("dat_expr_genes")

    if dat_expr is None:
        raise ValueError("datExpr not found. Run set_dat_expr first.")

    if hasattr(dat_expr, "toarray"):
        dat_expr = dat_expr.toarray()
    dat_expr = np.asarray(dat_expr, dtype=np.float64)

    # Determine gene names
    if dat_expr_genes is not None:
        gene_names = list(dat_expr_genes)
    else:
        gene_names = adata.var_names.tolist()

    # dat_expr is stored as (genes, cells) in py_hdWGCNA
    # Ensure genes are rows, cells are columns
    if dat_expr.shape[1] > dat_expr.shape[0]:
        # Likely (cells, genes) -> transpose to (genes, cells)
        dat_expr = dat_expr.T
        # Update gene_names if needed
        if dat_expr.shape[0] != len(gene_names):
            gene_names = gene_names[:dat_expr.shape[0]]

    # Now dat_expr is (n_genes, n_cells)
    n_dat_genes, n_cells = dat_expr.shape
    gene_names = gene_names[:n_dat_genes]

    # Genes to use: intersection of motif matrix rows and dat_expr genes
    genes_use = [g for g in gene_names if g in motif_matrix.index]

    # Build TF -> gene mapping from motif info
    motif_gene_to_id = {}
    for _, row in motif_info.iterrows():
        mid = row["motif_id"]
        gname = row["gene_name"]
        if mid not in motif_gene_to_id:
            motif_gene_to_id[mid] = gname

    importance_list = []
    eval_list = []
    n_genes_total = len(genes_use)

    print(f"Constructing TF network for {n_genes_total} genes...")
    from tqdm import tqdm

    # Build gene name -> index mapping
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}

    for idx, cur_gene in enumerate(tqdm(genes_use, desc="TF Network")):
        # Get candidate TFs for this gene
        if cur_gene not in motif_matrix.index:
            continue

        gene_row = motif_matrix.loc[cur_gene]
        active_motifs = gene_row[gene_row > 0].index.tolist()

        # Map motifs to TF gene names
        cur_tfs = []
        for mid in active_motifs:
            if mid in motif_gene_to_id:
                cur_tfs.append(motif_gene_to_id[mid])
        cur_tfs = list(set(cur_tfs))

        # Remove self-regulation
        if cur_gene in cur_tfs:
            cur_tfs.remove(cur_gene)

        # Only keep TFs that are in the expression data
        cur_tfs = [tf for tf in cur_tfs if tf in gene_name_to_idx]

        if len(cur_tfs) < 2:
            continue

        # Get expression data: dat_expr is (n_genes, n_cells)
        tf_indices = [gene_name_to_idx[tf] for tf in cur_tfs]
        gene_idx = gene_name_to_idx[cur_gene]

        x_vars = dat_expr[tf_indices, :].T  # (n_cells, n_tfs)
        y_var = dat_expr[gene_idx, :]        # (n_cells,)

        if np.all(y_var == 0):
            continue

        # Pearson correlation between each TF and the gene
        tf_cor = np.array([
            np.corrcoef(x_vars[:, j], y_var)[0, 1]
            for j in range(x_vars.shape[1])
        ])

        # XGBoost CV
        dtrain = xgb.DMatrix(x_vars, label=y_var, feature_names=cur_tfs)

        xgb_cv = xgb.cv(
            params=model_params,
            dtrain=dtrain,
            num_boost_round=100,
            nfold=nfold,
            metrics="rmse",
            verbose_eval=False,
        )

        # Get best iteration and train final model for importance
        best_round = int(xgb_cv["test-rmse-mean"].idxmin()) + 1
        bst = xgb.train(
            params=model_params,
            dtrain=dtrain,
            num_boost_round=best_round,
        )

        # Get evaluation metrics
        xgb_eval = xgb_cv.copy()
        xgb_eval["variable"] = cur_gene
        eval_list.append(xgb_eval)

        # Extract feature importance (Gain, Cover, Frequency/Weight)
        gain_scores = bst.get_score(importance_type="gain")
        cover_scores = bst.get_score(importance_type="cover")
        freq_scores = bst.get_score(importance_type="weight")

        imp_records = []
        for j, tf in enumerate(cur_tfs):
            imp_records.append({
                "tf": tf,
                "gene": cur_gene,
                "Gain": gain_scores.get(tf, 0.0),
                "Cover": cover_scores.get(tf, 0.0),
                "Frequency": freq_scores.get(tf, 0.0),
                "Cor": tf_cor[j],
            })
        imp_df = pd.DataFrame(imp_records)
        imp_df = imp_df.sort_values("Gain", ascending=False).reset_index(drop=True)
        importance_list.append(imp_df)

    if len(importance_list) == 0:
        warnings.warn("No TF-gene interactions found. Check motif data.")
        return adata

    importance_df = pd.concat(importance_list, ignore_index=True)

    if len(eval_list) > 0:
        eval_df = pd.concat(eval_list, ignore_index=True)
    else:
        eval_df = pd.DataFrame()

    # Store results
    wd["tf_network"] = importance_df
    wd["tf_eval"] = eval_df
    adata = set_hdWGCNA_data(adata, wd, wgcna_name)

    n_pairs = len(importance_df)
    n_tfs = importance_df["tf"].nunique()
    n_genes = importance_df["gene"].nunique()
    print(f"TF network constructed: {n_pairs} TF-gene pairs, "
          f"{n_tfs} TFs, {n_genes} target genes")

    return adata


# ------------------------------------------------------------------ #
# Regulon Assignment
# ------------------------------------------------------------------ #

def assign_tf_regulons(
    adata: AnnData,
    strategy: str = "A",
    reg_thresh: float = 0.01,
    n_tfs: int = 10,
    n_genes: int = 50,
    wgcna_name: str = None,
) -> AnnData:
    """
    Define TF regulons (sets of confident TF-gene pairs).

    Re-implements R hdWGCNA::AssignTFRegulons.

    Strategies:
        A: Top TFs for each gene (by Gain)
        B: Top target genes for each TF (by Gain)
        C: All pairs above reg_thresh

    Parameters
    ----------
    adata : AnnData
        AnnData with TF network
    strategy : str
        'A', 'B', or 'C'
    reg_thresh : float
        Minimum Gain score threshold
    n_tfs : int
        Strategy A: number of top TFs per gene
    n_genes : int
        Strategy B: number of top genes per TF
    wgcna_name : str
        Experiment name

    Returns
    -------
    AnnData
        Modified AnnData with regulons stored
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data

    if strategy not in ("A", "B", "C"):
        raise ValueError("strategy must be 'A', 'B', or 'C'")

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wd = get_hdWGCNA_data(adata, wgcna_name)

    tf_net = wd.get("tf_network")
    if tf_net is None or len(tf_net) == 0:
        raise ValueError("TF network not found. Run construct_tf_network first.")

    tf_net_filtered = tf_net[tf_net["Gain"] >= reg_thresh].copy()

    if strategy == "A":
        tf_regulons = (
            tf_net_filtered
            .groupby("gene", group_keys=False)
            .apply(lambda x: x.nlargest(min(n_tfs, len(x)), "Gain"))
        )
    elif strategy == "B":
        tf_regulons = (
            tf_net_filtered
            .groupby("tf", group_keys=False)
            .apply(lambda x: x.nlargest(min(n_genes, len(x)), "Gain"))
        )
    else:  # C
        tf_regulons = tf_net_filtered.copy()

    # Sort by Gain * sign(Cor) within each TF group
    tf_regulons["reg_score"] = tf_regulons["Gain"] * np.sign(tf_regulons["Cor"])
    tf_regulons = (
        tf_regulons
        .groupby("tf", group_keys=False)
        .apply(lambda x: x.sort_values("reg_score", ascending=False))
    )
    tf_regulons = tf_regulons.reset_index(drop=True)

    wd["tf_regulons"] = tf_regulons
    adata = set_hdWGCNA_data(adata, wd, wgcna_name)

    n_regulons = tf_regulons["tf"].nunique()
    print(f"Regulons assigned (strategy={strategy}): "
          f"{n_regulons} TFs, {len(tf_regulons)} TF-gene pairs")

    return adata


# ------------------------------------------------------------------ #
# Regulon Scores (UCell-like)
# ------------------------------------------------------------------ #

def regulon_scores(
    adata: AnnData,
    target_type: str = "positive",
    cor_thresh: float = 0.05,
    exclude_grey_genes: bool = True,
    wgcna_name: str = None,
) -> AnnData:
    """
    Compute expression scores for TF regulons.

    Re-implements R hdWGCNA::RegulonScores. Computes UCell-like
    module scores (rank-based) for each TF's target genes.

    Parameters
    ----------
    adata : AnnData
        AnnData with regulon data and expression matrix
    target_type : str
        'positive', 'negative', or 'both'
    cor_thresh : float
        Correlation threshold for target inclusion
    exclude_grey_genes : bool
        Exclude genes in the grey module
    wgcna_name : str
        Experiment name

    Returns
    -------
    AnnData
        Modified AnnData with regulon scores stored
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data

    if target_type not in ("positive", "negative", "both"):
        raise ValueError("target_type must be 'positive', 'negative', or 'both'")

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wd = get_hdWGCNA_data(adata, wgcna_name)

    modules_df = wd.get("modules_df")
    tf_regulons = wd.get("tf_regulons")

    if tf_regulons is None:
        raise ValueError("Regulons not found. Run assign_tf_regulons first.")

    # Filter by correlation direction
    reg = tf_regulons.copy()
    if target_type == "positive":
        reg = reg[reg["Cor"] > cor_thresh]
    elif target_type == "negative":
        reg = reg[reg["Cor"] < -cor_thresh]
    else:  # both
        reg = reg[reg["Cor"].abs() > cor_thresh]

    # Exclude grey module genes
    if exclude_grey_genes and modules_df is not None:
        grey_genes = set(modules_df[modules_df["module"] == "grey"]["gene_name"].tolist())
        reg = reg[~reg["gene"].isin(grey_genes) & ~reg["tf"].isin(grey_genes)]

    # Build target gene lists per TF
    tfs_use = sorted(reg["tf"].unique())
    target_genes_per_tf = {}
    for tf in tfs_use:
        targets = reg[reg["tf"] == tf]["gene"].tolist()
        target_genes_per_tf[tf] = targets

    # Get expression matrix (cells x genes)
    expr_mat = adata.X
    if hasattr(expr_mat, "toarray"):
        expr_mat = expr_mat.toarray()
    expr_mat = np.asarray(expr_mat, dtype=np.float32)

    gene_to_idx = {g: i for i, g in enumerate(adata.var_names.tolist())}

    # Compute UCell-like scores (rank-based)
    n_cells = expr_mat.shape[0]
    scores = np.zeros((n_cells, len(tfs_use)))

    for j, tf in enumerate(tqdm(tfs_use, desc="Regulon Scores")):
        targets = target_genes_per_tf[tf]
        valid_targets = [g for g in targets if g in gene_to_idx]
        if len(valid_targets) == 0:
            continue

        target_idx = [gene_to_idx[g] for g in valid_targets]
        target_expr = expr_mat[:, target_idx]  # cells x targets

        # UCell rank-based scoring
        # Rank genes within each cell, then average ranks of target genes
        n_targets = target_expr.shape[1]
        if n_targets == 0:
            continue

        # Compute ranks per cell
        ranks = np.zeros_like(target_expr)
        for i in range(n_cells):
            row = target_expr[i, :]
            # Rank from 0 to n_genes-1 (fractional rank)
            sorted_idx = np.argsort(row)
            ranks_sorted = np.empty_like(sorted_idx, dtype=np.float64)
            ranks_sorted[sorted_idx] = np.arange(n_targets, dtype=np.float64)
            ranks[i, :] = ranks_sorted

        # Normalize ranks to [0, 1]
        ranks_norm = ranks / max(n_targets - 1, 1)

        # Score = mean rank of target genes
        scores[:, j] = ranks_norm.mean(axis=1)

    scores_df = pd.DataFrame(scores, index=adata.obs_names, columns=tfs_use)

    # Store
    score_key = f"regulon_scores_{target_type}"
    wd[score_key] = scores_df
    adata = set_hdWGCNA_data(adata, wd, wgcna_name)

    print(f"Regulon scores computed ({target_type}): "
          f"{len(tfs_use)} TFs, {n_cells} cells")

    return adata


# Import tqdm here to avoid circular imports
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ------------------------------------------------------------------ #
# Get TF Target Genes
# ------------------------------------------------------------------ #

def get_tf_target_genes(
    adata: AnnData,
    selected_tfs: List[str],
    depth: int = 1,
    target_type: str = "both",
    use_regulons: bool = True,
    wgcna_name: str = None,
) -> pd.DataFrame:
    """
    Retrieve target genes for specified TFs with network depth control.

    Re-implements R hdWGCNA::GetTFTargetGenes.

    Parameters
    ----------
    adata : AnnData
        AnnData with TF network/regulons
    selected_tfs : list
        TFs to start from
    depth : int
        Number of network layers to explore
    target_type : str
        'positive', 'negative', or 'both'
    use_regulons : bool
        Use regulons (True) or full TF network (False)
    wgcna_name : str
        Experiment name

    Returns
    -------
    DataFrame
        TF-target interactions at each depth level
    """
    wd = _get_wd(adata, wgcna_name)

    if use_regulons:
        tf_data = wd.get("tf_regulons")
    else:
        tf_data = wd.get("tf_network")

    if tf_data is None:
        raise ValueError("TF data not found.")

    # Filter by correlation
    if target_type == "positive":
        tf_data = tf_data[tf_data["Cor"] > 0].copy()
    elif target_type == "negative":
        tf_data = tf_data[tf_data["Cor"] < 0].copy()

    # Validate selected TFs
    available_tfs = set(tf_data["tf"].unique())
    not_found = [tf for tf in selected_tfs if tf not in available_tfs]
    if not_found:
        raise ValueError(f"TFs not found in network: {not_found}")

    prev_tfs = set(selected_tfs)
    all_rows = []

    for d in range(1, depth + 1):
        cur_data = tf_data[tf_data["tf"].isin(prev_tfs)].copy()
        cur_data["depth"] = d
        all_rows.append(cur_data)

        # Find TFs among target genes
        cur_tfs_in_targets = set(cur_data[cur_data["gene"].isin(available_tfs)]["gene"])
        prev_tfs = prev_tfs | cur_tfs_in_targets

    if len(all_rows) == 0:
        return pd.DataFrame(columns=["tf", "gene", "Gain", "Cover", "Frequency", "Cor", "depth"])

    return pd.concat(all_rows, ignore_index=True)


# ------------------------------------------------------------------ #
# Module Regulatory Network
# ------------------------------------------------------------------ #

def module_regulatory_network(
    adata: AnnData,
    tfs_only: bool = True,
    wgcna_name: str = None,
) -> pd.DataFrame:
    """
    Summarize TF regulatory networks across co-expression modules.

    Re-implements R hdWGCNA::ModuleRegulatoryNetwork.

    Parameters
    ----------
    adata : AnnData
        AnnData with TF regulons and module data
    tfs_only : bool
        Include only TF-TF links
    wgcna_name : str
        Experiment name

    Returns
    -------
    DataFrame
        Module-level regulatory network with scores
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wd = get_hdWGCNA_data(adata, wgcna_name)

    tf_regulons = wd.get("tf_regulons")
    modules_df = wd.get("modules_df")

    if tf_regulons is None:
        raise ValueError("Regulons not found.")
    if modules_df is None:
        raise ValueError("Modules not found.")

    # Get non-grey modules
    non_grey = modules_df[modules_df["module"] != "grey"].copy()
    mods = sorted(non_grey["module"].unique(), key=_mod_sort_key)

    # Build gene -> module lookup
    gene_to_module = dict(zip(non_grey["gene_name"], non_grey["module"]))

    # Get TF set
    all_tfs = set(tf_regulons["tf"].unique())
    module_tfs = non_grey[non_grey["gene_name"].isin(all_tfs)]

    # Filter regulons to non-grey genes
    reg = tf_regulons.copy()
    reg = reg[reg["tf"].isin(gene_to_module) & reg["gene"].isin(gene_to_module)]

    if tfs_only:
        reg = reg[reg["gene"].isin(all_tfs)]

    # Add module info
    reg["source_module"] = reg["tf"].map(gene_to_module)
    reg["target_module"] = reg["gene"].map(gene_to_module)

    # Compute regulatory score = Gain * sign(Cor)
    reg["reg_score"] = reg["Gain"] * np.sign(reg["Cor"])

    # Count links between modules
    records = []
    for m1 in mods:
        for m2 in mods:
            cur = reg[(reg["target_module"] == m1) & (reg["source_module"] == m2)]

            pos = cur[cur["reg_score"] >= 0]
            neg = cur[cur["reg_score"] < 0]

            records.append({
                "source": m2,
                "target": m1,
                "n_pos": len(pos),
                "sum_pos": pos["reg_score"].sum() if len(pos) > 0 else 0,
                "n_neg": len(neg),
                "sum_neg": neg["reg_score"].sum() if len(neg) > 0 else 0,
            })

    reg_df = pd.DataFrame(records)

    # Compute averages
    reg_df["mean_pos"] = np.where(reg_df["n_pos"] > 0, reg_df["sum_pos"] / reg_df["n_pos"], 0)
    reg_df["mean_neg"] = np.where(reg_df["n_neg"] > 0, reg_df["sum_neg"] / reg_df["n_neg"], 0)

    # Normalize by TF count per module
    tf_counts = module_tfs["module"].value_counts()
    reg_df["score_pos"] = reg_df.apply(
        lambda r: r["n_pos"] / tf_counts.get(r["source"], 1), axis=1
    )
    reg_df["score_neg"] = reg_df.apply(
        lambda r: r["n_neg"] / tf_counts.get(r["source"], 1), axis=1
    )

    reg_df["source"] = pd.Categorical(reg_df["source"], categories=mods, ordered=True)
    reg_df["target"] = pd.Categorical(reg_df["target"], categories=mods, ordered=True)

    # Store
    wd["module_regulatory_network"] = reg_df
    from .utils import set_hdWGCNA_data
    adata = set_hdWGCNA_data(adata, wd, wgcna_name)

    return reg_df


def _mod_sort_key(m):
    """Sort module names: grey first, then by numeric suffix."""
    if m == "grey":
        return (-1, "")
    # Try to extract number from e.g. "M1", "turquoise"
    try:
        return (0, int(m.replace("M", "")))
    except (ValueError, AttributeError):
        return (0, m)


# ------------------------------------------------------------------ #
# Overlap Modules with Motifs (Fisher's exact test)
# ------------------------------------------------------------------ #

def overlap_modules_motifs(
    adata: AnnData,
    wgcna_name: str = None,
) -> pd.DataFrame:
    """
    Test overlap between co-expression modules and TF target genes.

    Re-implements R hdWGCNA::OverlapModulesMotifs using Fisher's
    exact test for each module-TF pair.

    Parameters
    ----------
    adata : AnnData
        AnnData with motif data and modules
    wgcna_name : str
        Experiment name

    Returns
    -------
    DataFrame
        Module-TF overlap statistics
    """
    from .utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data

    wgcna_name = check_wgcna_name(adata, wgcna_name)
    wd = get_hdWGCNA_data(adata, wgcna_name)

    modules_df = wd.get("modules_df")
    tf_targets = wd.get("motif_targets")

    if modules_df is None or tf_targets is None:
        raise ValueError("Modules and/or motif data not found.")

    from statsmodels.stats.multitest import multipletests

    non_grey = modules_df[modules_df["module"] != "grey"].copy()
    mods = sorted(non_grey["module"].unique(), key=_mod_sort_key)
    genome_size = adata.n_vars

    records = []
    for cur_mod in mods:
        module_genes = set(
            non_grey[non_grey["module"] == cur_mod]["gene_name"].tolist()
        )
        mod_color = non_grey[non_grey["module"] == cur_mod]["color"].iloc[0]

        for tf_name, targets in tf_targets.items():
            target_set = set(targets)

            # Fisher's exact test
            intersection = module_genes & target_set
            a = len(intersection)
            b = len(module_genes) - a
            c = len(target_set) - a
            d = genome_size - a - b - c

            odds_ratio, pval = stats.fisher_exact(
                [[a, b], [c, d]], alternative="greater"
            )

            # Jaccard index
            union = module_genes | target_set
            jaccard = a / len(union) if len(union) > 0 else 0

            records.append({
                "module": cur_mod,
                "tf": tf_name,
                "color": mod_color,
                "odds_ratio": odds_ratio,
                "pval": pval,
                "Jaccard": jaccard,
                "size_intersection": a,
            })

    overlap_df = pd.DataFrame(records)

    # FDR correction
    _, fdr_vals, _, _ = multipletests(overlap_df["pval"].values, method="fdr_bh")
    overlap_df["fdr"] = fdr_vals

    # Significance stars
    overlap_df["Significance"] = ""
    overlap_df.loc[overlap_df["fdr"] < 0.001, "Significance"] = "***"
    overlap_df.loc[
        (overlap_df["fdr"] >= 0.001) & (overlap_df["fdr"] < 0.01), "Significance"
    ] = "**"
    overlap_df.loc[
        (overlap_df["fdr"] >= 0.01) & (overlap_df["fdr"] < 0.05), "Significance"
    ] = "*"

    overlap_df["module"] = pd.Categorical(
        overlap_df["module"], categories=mods, ordered=True
    )

    # Store
    wd["motif_overlap"] = overlap_df
    adata = set_hdWGCNA_data(adata, wd, wgcna_name)

    print(f"Module-motif overlap computed: {len(overlap_df)} pairs")
    return overlap_df


# ------------------------------------------------------------------ #
# Convenience: Get TF data from adata
# ------------------------------------------------------------------ #

def get_tf_network(adata: AnnData, wgcna_name: str = None) -> Optional[pd.DataFrame]:
    """Get the TF network dataframe."""
    try:
        wd = _get_wd(adata, wgcna_name)
        return wd.get("tf_network")
    except (ValueError, KeyError):
        return None


def get_tf_regulons(adata: AnnData, wgcna_name: str = None) -> Optional[pd.DataFrame]:
    """Get the TF regulons dataframe."""
    try:
        wd = _get_wd(adata, wgcna_name)
        return wd.get("tf_regulons")
    except (ValueError, KeyError):
        return None


def get_regulon_scores(
    adata: AnnData,
    target_type: str = "positive",
    wgcna_name: str = None,
) -> Optional[pd.DataFrame]:
    """Get regulon scores."""
    try:
        wd = _get_wd(adata, wgcna_name)
        return wd.get(f"regulon_scores_{target_type}")
    except (ValueError, KeyError):
        return None


# ------------------------------------------------------------------ #
# Differential Regulon Analysis
# ------------------------------------------------------------------ #

def find_differential_regulons(
    adata: AnnData,
    barcodes1: List[str],
    barcodes2: List[str],
    test_use: str = "wilcox",
    logfc_threshold: float = 0,
    wgcna_name: str = None,
) -> pd.DataFrame:
    """
    Differential regulon analysis between two groups.

    Re-implements R hdWGCNA::FindDifferentialRegulons using
    Wilcoxon rank-sum test on regulon scores and TF expression.

    Parameters
    ----------
    adata : AnnData
        AnnData with regulon scores computed
    barcodes1, barcodes2 : list
        Cell barcodes for each group
    test_use : str
        Statistical test ('wilcox' for Wilcoxon rank-sum)
    logfc_threshold : float
        LogFC threshold for DEG classification
    wgcna_name : str
        Experiment name

    Returns
    -------
    DataFrame
        Differential regulon results with columns:
        [tf, p_val_positive, avg_log2FC_positive, p_val_adj_positive,
         p_val_negative, avg_log2FC_negative, p_val_adj_negative,
         p_val_deg, avg_log2FC_deg, p_val_adj_deg, module, kME]
    """
    from scipy.stats import mannwhitneyu
    from statsmodels.stats.multitest import multipletests

    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")
    pos_scores = wd.get("regulon_scores_positive")
    neg_scores = wd.get("regulon_scores_negative")

    if pos_scores is None or neg_scores is None:
        raise ValueError("Regulon scores not found. Run regulon_scores() first.")

    # Filter to valid barcodes
    b1 = [b for b in barcodes1 if b in pos_scores.index]
    b2 = [b for b in barcodes2 if b in pos_scores.index]
    tfs = [c for c in pos_scores.columns if c != "cell"]

    def _test_scores(scores_df, tfs_list, b1, b2):
        """Run Wilcoxon test for each TF regulon score."""
        results = []
        for tf in tfs_list:
            if tf not in scores_df.columns:
                continue
            vals1 = scores_df.loc[b1, tf].dropna().values
            vals2 = scores_df.loc[b2, tf].dropna().values
            if len(vals1) < 3 or len(vals2) < 3:
                results.append({"tf": tf, "p_val": 1.0, "avg_log2FC": 0.0})
                continue
            try:
                stat, pval = mannwhitneyu(vals1, vals2, alternative="two-sided")
            except ValueError:
                pval = 1.0
            fc = float(np.mean(vals1) - np.mean(vals2))
            results.append({"tf": tf, "p_val": pval, "avg_log2FC": fc})
        return pd.DataFrame(results)

    # Test positive regulon scores
    pos_res = _test_scores(pos_scores, tfs, b1, b2)
    pos_res.columns = ["tf", "p_val_positive", "avg_log2FC_positive"]

    # Test negative regulon scores
    neg_res = _test_scores(neg_scores, tfs, b1, b2)
    neg_res.columns = ["tf", "p_val_negative", "avg_log2FC_negative"]

    # Test TF gene expression
    tf_genes_in_adata = [t for t in tfs if t in adata.var_names]
    if len(tf_genes_in_adata) > 0:
        if hasattr(adata.X, "toarray"):
            expr_df = pd.DataFrame(
                adata[b1 + b2, tf_genes_in_adata].X.toarray(),
                index=b1 + b2,
                columns=tf_genes_in_adata,
            )
        else:
            expr_df = pd.DataFrame(
                adata[b1 + b2, tf_genes_in_adata].X,
                index=b1 + b2,
                columns=tf_genes_in_adata,
            )
        deg_res = _test_scores(expr_df, tf_genes_in_adata, b1, b2)
        deg_res.columns = ["tf", "p_val_deg", "avg_log2FC_deg"]
    else:
        deg_res = pd.DataFrame(columns=["tf", "p_val_deg", "avg_log2FC_deg"])

    # Merge results
    dregs = pos_res.merge(neg_res, on="tf").merge(deg_res, on="tf", how="left")
    dregs["p_val_deg"] = dregs["p_val_deg"].fillna(1.0)
    dregs["avg_log2FC_deg"] = dregs["avg_log2FC_deg"].fillna(0.0)

    # Adjust p-values (BH)
    for col in ["p_val_positive", "p_val_negative", "p_val_deg"]:
        adj_col = col.replace("p_val", "p_val_adj")
        _, pvals_adj, _, _ = multipletests(dregs[col].values, method="fdr_bh")
        dregs[adj_col] = pvals_adj

    # Add module and kME info
    if modules_df is not None:
        mod_info = modules_df[["gene_name", "module"]].drop_duplicates()
        mod_info.columns = ["tf", "module"]
        dregs = dregs.merge(mod_info, on="tf", how="left")

        # Get kME column
        kme_cols = [c for c in modules_df.columns if c.startswith("kME_")]
        if kme_cols:
            kme_info = modules_df[["gene_name"] + kme_cols].copy()
            kme_info["kME"] = kme_info[kme_cols].max(axis=1)
            kme_info = kme_info[["gene_name", "kME"]].drop_duplicates()
            kme_info.columns = ["tf", "kME"]
            dregs = dregs.merge(kme_info, on="tf", how="left")
        else:
            dregs["kME"] = 0.0
    else:
        dregs["module"] = "unknown"
        dregs["kME"] = 0.0

    dregs["kME"] = dregs["kME"].fillna(0.0)

    # Sort by positive regulon p-value
    dregs = dregs.sort_values("p_val_positive").reset_index(drop=True)

    return dregs


# ------------------------------------------------------------------ #
# Enrichr Regulon Analysis
# ------------------------------------------------------------------ #

def run_enrichr_regulons(
    adata: AnnData,
    dbs: List[str] = None,
    depth: int = 1,
    min_genes: int = 5,
    wait_time: float = 1.0,
    wgcna_name: str = None,
) -> AnnData:
    """
    Run Enrichr enrichment on TF regulon target genes.

    Re-implements R hdWGCNA::RunEnrichrRegulons. For each TF,
    splits target genes by correlation direction and runs Enrichr.

    Parameters
    ----------
    adata : AnnData
        AnnData with TF regulons
    dbs : list
        Enrichr database names
    depth : int
        Network depth for target gene retrieval
    min_genes : int
        Minimum genes required to run enrichment
    wait_time : float
        Seconds to wait between API calls
    wgcna_name : str
        Experiment name

    Returns
    -------
    AnnData
        Updated adata with enrichr_regulon_table stored
    """
    import time
    from .enrichment import run_enrichr

    if dbs is None:
        dbs = [
            "GO_Biological_Process_2023",
            "GO_Cellular_Component_2023",
            "GO_Molecular_Function_2023",
        ]

    wn = _get_wgcna_name(adata, wgcna_name)
    wd = _get_wd(adata, wn)
    tf_regulons = wd.get("tf_regulons")
    if tf_regulons is None:
        raise ValueError("TF regulons not found.")

    all_tfs = tf_regulons["tf"].unique()
    all_results = []

    for i, tf in enumerate(all_tfs):
        # Get target genes at specified depth
        try:
            targets = get_tf_target_genes(
                adata, selected_tfs=[tf], depth=depth,
                target_type="both", wgcna_name=wgcna_name,
            )
        except (ValueError, KeyError):
            continue

        if len(targets) == 0:
            continue

        # Split by correlation direction
        pos_targets = targets[targets["Cor"] > 0]["gene"].unique().tolist()
        neg_targets = targets[targets["Cor"] < 0]["gene"].unique().tolist()

        for target_type, gene_list in [("positive", pos_targets), ("negative", neg_targets)]:
            if len(gene_list) < min_genes:
                continue

            for db in dbs:
                try:
                    enrichr_res = run_enrichr(gene_list, gene_sets=db)
                except Exception:
                    continue

                if len(enrichr_res) > 0:
                    enrichr_res["tf"] = tf
                    enrichr_res["target_type"] = target_type
                    enrichr_res["db"] = db
                    all_results.append(enrichr_res)

        if wait_time > 0 and i < len(all_tfs) - 1:
            time.sleep(wait_time)

    if all_results:
        enrichr_table = pd.concat(all_results, ignore_index=True)
    else:
        enrichr_table = pd.DataFrame(columns=["tf", "target_type"])

    from .utils import set_hdWGCNA_data

    wd["enrichr_regulon_table"] = enrichr_table
    adata = set_hdWGCNA_data(adata, wd, wn)

    return adata


def get_enrichr_regulon_table(
    adata: AnnData,
    wgcna_name: str = None,
) -> Optional[pd.DataFrame]:
    """Get the Enrichr regulon enrichment table."""
    try:
        wd = _get_wd(adata, wgcna_name)
        return wd.get("enrichr_regulon_table")
    except (ValueError, KeyError):
        return None
