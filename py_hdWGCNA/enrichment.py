"""
Enrichment functions for hdWGCNA using Enrichr API.

Pure-computation module: run_enrichr, run_enrichr_modules.
Plotting functions are in plotting.py (enrichr_bar_plot, enrichr_dot_plot).
"""

from __future__ import annotations

import time
import warnings
from typing import List, Union, Dict
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


def run_enrichr(
    gene_list: Union[List[str], np.ndarray],
    gene_sets: str = "GO_Biological_Process_2023",
    species: str = "human",
) -> pd.DataFrame:
    """
    Run Enrichr enrichment analysis on a single gene list.

    Replicates R's Enrichr function.

    Parameters
    ----------
    gene_list : list or array
        Gene symbols to analyze
    gene_sets : str
        Enrichr library name (e.g., 'GO_Biological_Process_2023')
    species : str
        Species ('human' or 'mouse')

    Returns
    -------
    DataFrame with enrichment results
    """
    try:
        import requests
    except ImportError:
        raise ImportError("requests required. Install with: pip install requests")

    genes = list(gene_list)
    if len(genes) == 0:
        warnings.warn("Empty gene list provided.", UserWarning)
        return pd.DataFrame()

    base_url = "https://maayanlab.cloud/Enrichr/enrich"
    _query_string = "?userListId=%s&backgroundType=%s"

    try:
        payload = {
            "list": (None, ",".join(genes)),
            "description": (None, "py-hdWGCNA enrichment"),
            "species": (None, species),
        }
        response = requests.post(base_url + "AddList", files=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        user_list_id = result.get("userListId", None)

        if user_list_id is None:
            warnings.warn("Enrichr did not return userListId.", UserWarning)
            return pd.DataFrame()

        time.sleep(0.8)

        response = requests.get(base_url % (user_list_id, gene_sets), timeout=60)
        response.raise_for_status()
        data = response.json().get(gene_sets, [])

        if not data:
            warnings.warn(f"No results returned for {gene_sets}.", UserWarning)
            return pd.DataFrame()

        df = pd.DataFrame(data)
        col_rename = {
            0: "rank",
            1: "term",
            2: "pvalue",
            3: "zscore",
            4: "combined_score",
            5: "overlap_genes",
            6: "n_genes",
            7: "pvalue_bonferroni",
            8: "pvalue_fisher",
            9: "pvalue_ease",
        }
        df = df.rename(
            columns={
                df.columns[i]: col_rename[i] for i in range(min(len(df.columns), 10))
            }
        )

        for col in ["pvalue", "combined_score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        overlap_col = None
        for c in df.columns:
            if "overlap" in c.lower():
                overlap_col = c
                break

        if overlap_col is not None:
            df["overlap"] = df[overlap_col].apply(
                lambda x: len(str(x).split(";")) if pd.notna(x) else 0
            )
        else:
            df["overlap"] = 0

        if "n_genes" in df.columns:
            df["n_genes"] = pd.to_numeric(df["n_genes"], errors="coerce").fillna(
                len(genes)
            )

        df = df.sort_values("pvalue").reset_index(drop=True)

        return (
            df[["term", "pvalue", "overlap", "n_genes", "combined_score"]]
            if all(
                c in df.columns
                for c in ["term", "pvalue", "overlap", "n_genes", "combined_score"]
            )
            else df
        )

    except Exception as e:
        warnings.warn(f"Enrichr API error: {e}", UserWarning)
        return pd.DataFrame()


def run_enrichr_modules(
    adata: AnnData,
    gene_sets: List[str] = None,
    exclude_grey: bool = True,
    species: str = "human",
    wgcna_name: str = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run Enrichr enrichment for each non-grey module.

    Replicates R's EnrichrModules function.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    gene_sets : list
        Enrichr libraries to query
    exclude_grey : bool
        Exclude grey module
    species : str
        Species
    wgcna_name : str
        Experiment name

    Returns
    -------
    Dict mapping module name to enrichment DataFrame
    """
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")

    if modules_df is None:
        raise ValueError("No module data found.")

    if gene_sets is None:
        gene_sets = [
            "GO_Biological_Process_2023",
            "KEGG_2021_Human",
            "Reactome_2022",
            "WikiPathway_2023_Human",
        ]

    mods_df = modules_df.copy()
    if exclude_grey:
        mods_df = mods_df[mods_df["module"] != "grey"]

    unique_mods = sorted(mods_df["module"].unique())
    all_enrichr = {}

    for cur_mod in unique_mods:
        mod_genes = mods_df[mods_df["module"] == cur_mod]
        gene_list = (
            mod_genes["gene_name"].values.tolist()
            if "gene_name" in mod_genes.columns
            else mod_genes.index.tolist()
        )

        gene_list = [str(g).strip() for g in gene_list if pd.notna(g)]
        gene_list = list(dict.fromkeys(gene_list))

        if len(gene_list) < 2:
            print(f"Skipping {cur_mod}, too few genes ({len(gene_list)}).")
            continue

        print(f"Running Enrichr for {cur_mod} ({len(gene_list)} genes)...")

        mod_results = {}
        for gs in gene_sets:
            res = run_enrichr(gene_list, gene_sets=gs, species=species)
            if len(res) > 0:
                res["database"] = gs
                res["module"] = cur_mod
                mod_results[gs] = res

        if mod_results:
            combined = pd.concat(mod_results.values(), ignore_index=True)
            all_enrichr[cur_mod] = combined
        else:
            print(f"  No significant results for {cur_mod}")

    return all_enrichr
