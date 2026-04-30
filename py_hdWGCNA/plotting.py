"""
Visualization functions for hdWGCNA analysis.

Pure-Python re-implementation of R hdWGCNA plotting functions using matplotlib.
All plots follow publication-quality standards: 300 DPI vector output, high text-to-canvas ratio.

Functions (14 total):
  Base plots (5):
    - plot_soft_powers, module_feature_plot, plot_dendrogram,
      plot_kmes, module_correlogram
  Network plots (3):
    - module_network_plot, hub_gene_network_plot, module_umap_plot
  DME plots (2):
    - plot_dmes_volcano, plot_dmes_lollipop
  Trait correlation (1):
    - plot_module_trait_correlation
  Enrichment plots (2):
    - enrichr_bar_plot, enrichr_dot_plot
  Preservation (1):
    - plot_module_preservation
"""

from __future__ import annotations

import os
from typing import List, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import leaves_list
from anndata import AnnData

try:
    import umap as umap_lib
except ImportError:
    umap_lib = None


# WGCNA standard color palette (same order as R)
WGCNA_COLORS = [
    "turquoise",
    "blue",
    "brown",
    "yellow",
    "green",
    "grey60",
    "magenta",
    "purple",
    "pink",
    "black",
    "red",
    "cyan",
    "greenyellow",
    "tan",
    "salmon",
    "skyblue",
    "orange",
    "midnightblue",
    "lightyellow",
    "lightcyan",
    "coral",
]


def _get_wgcna_name(adata: AnnData, wgcna_name: str = None) -> str:
    """Get active wgcna name."""
    if wgcna_name is None:
        wgcna_name = adata.uns.get("hdWGCNA", {}).get("active_wgcna", None)
    if wgcna_name is None:
        raise ValueError("No active hdWGCNA experiment found.")
    return wgcna_name


def _get_wd(adata: AnnData, wgcna_name: str = None) -> dict:
    """Get hdWGCNA experiment data dict."""
    wn = _get_wgcna_name(adata, wgcna_name)
    return adata.uns["hdWGCNA"][wn]


def _get_tom_similarity(wd: dict):
    """Get TOM similarity matrix from hdWGCNA data.

    R hdWGCNA's GetTOM returns consTomDS which stores TOM similarity
    (despite the DS suffix). The diagonal is 0 in R's output.
    This function detects whether the stored matrix is similarity or
    dissimilarity by checking the diagonal, and always returns
    TOM similarity matching R's convention (diagonal = 0, off-diagonal
    values in [0, ~0.34] range for typical signed networks).
    """
    TOM = wd.get("TOM")
    if TOM is None:
        return None, None

    if isinstance(TOM, pd.DataFrame):
        TOM_arr = TOM.values
        tom_genes = list(TOM.index)
    else:
        TOM_arr = np.asarray(TOM)
        tom_genes = wd.get("dat_expr_genes", list(range(TOM_arr.shape[0])))

    diag_mean = np.diag(TOM_arr).mean()
    if diag_mean < 0.5:
        TOM_arr = 1.0 - TOM_arr
        np.fill_diagonal(TOM_arr, 0.0)
        np.clip(TOM_arr, 0.0, 1.0, out=TOM_arr)

    return TOM_arr, tom_genes


def _setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.transparent": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )


_R_COLOR_MAP = {
    "grey90": "#E6E6E6",
    "grey80": "#CCCCCC",
    "grey70": "#B3B3B3",
    "grey60": "#999999",
    "grey50": "#808080",
    "grey40": "#666666",
    "grey30": "#4D4D4D",
    "grey20": "#333333",
    "grey10": "#1A1A1A",
    "greenyellow": "#ADFF2F",
    "midnightblue": "#191970",
    "lightyellow": "#FFFFE0",
    "lightcyan": "#E0FFFF",
}


def _to_mpl_color(color_str):
    if color_str in _R_COLOR_MAP:
        return _R_COLOR_MAP[color_str]
    try:
        mcolors.to_rgba(color_str)
        return color_str
    except ValueError:
        hex_color = mcolors.CSS4_COLORS.get(color_str.lower())
        if hex_color:
            return hex_color
        return "#999999"


# ======================================================================== #
# PlotSoftPowers - Soft threshold power selection plots
# ======================================================================== #


def plot_soft_powers(
    adata: AnnData,
    selected_power: int = None,
    point_size: float = 50,
    text_size: int = 8,
    plot_connectivity: bool = True,
    wgcna_name: str = None,
    save_path: str = None,
) -> Union[plt.Figure, List[plt.Figure]]:
    """
    Plot Soft Power Threshold results (Scale-free topology fit & connectivity).

    Replicates R's PlotSoftPowers function.

    Parameters
    ----------
    adata : AnnData
        AnnData object with hdWGCNA results
    selected_power : int
        Power to highlight (auto-detected if None)
    point_size : float
        Size of data points
    text_size : int
        Size of power labels on points
    plot_connectivity : bool
        Whether to include connectivity subplots
    wgcna_name : str
        Name of hdWGCNA experiment
    save_path : str
        Path to save figure (PDF/SVG/PNG)

    Returns
    -------
    Figure or list of Figures
    """
    _setup_publication_style()
    _wd = _get_wd(adata, wgcna_name)
    pt = _wd.get("power_table")

    if pt is None or len(pt) == 0:
        raise ValueError("No power table found. Run test_soft_powers() first.")

    if isinstance(pt, pd.DataFrame):
        powers = pt["Power"].values
        sft_rsq = pt["SFT.R.sq"].values
        mean_k = np.asarray(pt.get("mean.k.", pt.get("mean_k", np.zeros(len(powers)))))
        median_k = np.asarray(
            pt.get("median.k.", pt.get("median_k", np.zeros(len(powers))))
        )
        max_k = np.asarray(pt.get("max.k.", pt.get("max_k", np.zeros(len(powers)))))
    else:
        powers = np.array(list(pt.keys()))
        sft_rsq = np.array([pt[p]["SFT.R.sq"] for p in powers])
        mean_k = np.array([pt[p].get("mean.k.", 0) for p in powers])
        median_k = np.array([pt[p].get("median.k.", 0) for p in powers])
        max_k = np.array([pt[p].get("max.k.", 0) for p in powers])

    if selected_power is None:
        sp = _wd.get("selected_power")
        if sp is not None:
            selected_power = int(sp)
        else:
            valid_idx = np.where(sft_rsq >= 0.85)[0]
            if len(valid_idx) > 0:
                selected_power = int(powers[valid_idx[0]])
            else:
                selected_power = int(powers[-1])

    sel_idx = np.where(powers == selected_power)[0][0]
    sft_r_val = sft_rsq[sel_idx]
    mean_k_val = mean_k[sel_idx]

    _n_plots = 4 if plot_connectivity else 1  # noqa: F841
    fig_width = 3.0 if not plot_connectivity else 7.0
    fig_height = 3.0 if not plot_connectivity else 6.0

    if plot_connectivity:
        fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height))
        axes = axes.flatten()
    else:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        axes = [ax]

    ax1 = axes[0]
    ax1.axhspan(-1, 0.8, color="grey", alpha=0.3, zorder=0)
    ax1.axhline(y=sft_r_val, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.axvline(x=selected_power, color="black", linestyle="--", linewidth=1, alpha=0.7)

    _colors = ["white" if p != selected_power else "black" for p in powers]  # noqa: F841
    ax1.scatter(
        powers,
        sft_rsq,
        s=point_size,
        c="none",
        edgecolors="black",
        linewidths=0.8,
        zorder=5,
    )
    ax1.scatter([selected_power], [sft_r_val], s=point_size * 1.3, c="black", zorder=6)

    for i, (p, r) in enumerate(zip(powers, sft_rsq)):
        ax1.text(
            p,
            r,
            str(int(p)),
            ha="center",
            va="center",
            fontsize=text_size,
            fontweight="bold" if p == selected_power else "normal",
            color="white" if p == selected_power else "black",
            zorder=7,
        )

    ax1.set_ylim(0, 1)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_xlabel("Soft Power Threshold", fontsize=11)
    ax1.set_ylabel("Scale-free Topology Model Fit, R^2", fontsize=11)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    if plot_connectivity:
        ax2 = axes[1]
        ax2.axhline(y=mean_k_val, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax2.axvline(
            x=selected_power, color="black", linestyle="--", linewidth=1, alpha=0.7
        )
        ax2.scatter(
            powers, mean_k, s=point_size, c="none", edgecolors="black", linewidths=0.8
        )
        ax2.scatter([selected_power], [mean_k_val], s=point_size * 1.3, c="black")
        for p, k in zip(powers, mean_k):
            ax2.text(
                p,
                k,
                str(int(p)),
                ha="center",
                va="center",
                fontsize=text_size,
                color="white" if p == selected_power else "black",
            )
        ax2.set_xlabel("Soft Power Threshold", fontsize=11)
        ax2.set_ylabel("Mean Connectivity", fontsize=11)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax3 = axes[2]
        med_k_val = median_k[sel_idx]
        ax3.axhline(y=med_k_val, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax3.axvline(
            x=selected_power, color="black", linestyle="--", linewidth=1, alpha=0.7
        )
        ax3.scatter(
            powers, median_k, s=point_size, c="none", edgecolors="black", linewidths=0.8
        )
        ax3.scatter([selected_power], [med_k_val], s=point_size * 1.3, c="black")
        for p, k in zip(powers, median_k):
            ax3.text(
                p,
                k,
                str(int(p)),
                ha="center",
                va="center",
                fontsize=text_size,
                color="white" if p == selected_power else "black",
            )
        ax3.set_xlabel("Soft Power Threshold", fontsize=11)
        ax3.set_ylabel("Median Connectivity", fontsize=11)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        ax4 = axes[3]
        max_k_val = max_k[sel_idx]
        ax4.axhline(y=max_k_val, color="black", linestyle="--", linewidth=1, alpha=0.7)
        ax4.axvline(
            x=selected_power, color="black", linestyle="--", linewidth=1, alpha=0.7
        )
        ax4.scatter(
            powers, max_k, s=point_size, c="none", edgecolors="black", linewidths=0.8
        )
        ax4.scatter([selected_power], [max_k_val], s=point_size * 1.3, c="black")
        for p, k in zip(powers, max_k):
            ax4.text(
                p,
                k,
                str(int(p)),
                ha="center",
                va="center",
                fontsize=text_size,
                color="white" if p == selected_power else "black",
            )
        ax4.set_xlabel("Soft Power Threshold", fontsize=11)
        ax4.set_ylabel("Max Connectivity", fontsize=11)
        ax4.spines["top"].set_visible(False)
        ax4.spines["right"].set_visible(False)

    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.92)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# ModuleFeaturePlot - Module eigengenes on UMAP/tSNE
# ======================================================================== #


def module_feature_plot(
    adata: AnnData,
    module_names: List[str] = None,
    reduction: str = "umap",
    features: str = "hMEs",
    point_size: float = 3,
    alpha: float = 1.0,
    restrict_range: bool = True,
    wgcna_name: str = None,
    save_path: str = None,
    ncols: int = 3,
    order_points: bool = True,
) -> Union[plt.Figure, Dict[str, plt.Figure]]:
    """
    Plot module eigengenes as feature plots on dimensionality reduction.

    Replicates R's ModuleFeaturePlot function.

    Parameters
    ----------
    adata : AnnData
        AnnData object with hdWGCNA results and reductions
    module_names : list
        Specific modules to plot (default: all non-grey)
    reduction : str
        Reduction to use ('umap', 'tsne', 'pca')
    features : str
        Which MEs to use: 'hMEs', 'MEs', 'scores', 'average'
    point_size : float
        Point size
    alpha : float
        Point transparency
    restrict_range : bool
        Symmetrize color range around zero
    wgcna_name : str
        Name of hdWGCNA experiment
    save_path : str
        Save path
    ncols : int
        Number of columns in multi-panel figure
    order_points : bool
        Sort points by value before plotting (default: True, matches R behavior)

    Returns
    -------
    Figure or dict of Figures
    """
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)

    me_key = {
        "hMEs": "hMEs",
        "MEs": "MEs",
        "scores": "module_scores",
        "average": "avg_module_expr",
    }.get(features, "hMEs")
    MEs = wd.get(me_key)

    if MEs is None:
        raise ValueError(f"No {features} found. Run module_eigengenes() first.")

    if isinstance(MEs, pd.DataFrame):
        MEs = MEs.copy()
    elif isinstance(MEs, np.ndarray):
        mod_names = wd.get("module_names", [f"M{i}" for i in range(MEs.shape[1])])
        MEs = pd.DataFrame(MEs, columns=mod_names)
    else:
        raise TypeError(f"Unexpected MEs type: {type(MEs)}")

    modules_df = wd.get("modules_df")
    if modules_df is not None and isinstance(modules_df, pd.DataFrame):
        mod_colors_dict = dict(zip(modules_df["module"], modules_df["color"]))
    else:
        mod_colors_dict = {}

    if module_names is None:
        module_names = [m for m in MEs.columns if m.lower() != "grey"]
    else:
        module_names = [m for m in module_names if m in MEs.columns]

    if len(module_names) == 0:
        print("No modules to plot.")
        return None

    red_key = f"X_{reduction}"
    if red_key not in adata.obsm:
        available = [k for k in adata.obsm.keys() if k.startswith("X_")]
        raise ValueError(f"Reduction '{reduction}' not found. Available: {available}")

    coords = adata.obsm[red_key][:, :2]
    x_name, y_name = f"{reduction}1", f"{reduction}2"

    n_mods = len(module_names)
    actual_ncols = min(ncols, n_mods)
    nrows = (n_mods + actual_ncols - 1) // actual_ncols

    panel_w = 2.8
    panel_h = 2.4
    fig, axes = plt.subplots(
        nrows,
        actual_ncols,
        figsize=(actual_ncols * panel_w, nrows * panel_h),
        constrained_layout=True,
    )

    if nrows == 1 and actual_ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif actual_ncols == 1:
        axes = axes.reshape(-1, 1)

    plot_list = {}
    for idx, cur_mod in enumerate(module_names):
        row, col = idx // actual_ncols, idx % actual_ncols
        ax = axes[row, col]

        if cur_mod not in MEs.columns:
            continue

        vals = MEs[cur_mod].values
        if len(vals) == 0 or np.all(np.isnan(vals)):
            continue

        cur_color = mod_colors_dict.get(cur_mod, "#0072B2")

        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        plot_range = (vmin, vmax)
        if restrict_range:
            abs_max = max(abs(vmin), abs(vmax))
            vmin, vmax = -abs_max, abs_max
            plot_range = (vmin, vmax)

        plot_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "val": vals})

        from matplotlib.colors import LinearSegmentedColormap

        colors_stops = ["#B3B3B3", "#F2F2F2", cur_color]
        n_bins = 100
        cmap_custom = LinearSegmentedColormap.from_list(
            "module_gradient", colors_stops, N=n_bins
        )

        if order_points:
            sort_idx = np.argsort(plot_df["val"].values)
            plot_df = plot_df.iloc[sort_idx].reset_index(drop=True)

        ax.scatter(
            plot_df["x"],
            plot_df["y"],
            c=plot_df["val"],
            cmap=cmap_custom,
            s=point_size,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            edgecolors="none",
            rasterized=True,
        )

        ax.set_xlabel(x_name, fontsize=10)
        ax.set_ylabel(y_name, fontsize=10)
        ax.set_title(cur_mod, fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=9)

        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks([plot_range[0], plot_range[1]])
        cbar.set_ticklabels(["-", "+"])
        cbar.ax.tick_params(labelsize=8)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        plot_list[cur_mod] = ax

    total_cells = nrows * actual_ncols
    for idx in range(len(module_names), total_cells):
        row, col = idx // actual_ncols, idx % actual_ncols
        ax = axes[row, col] if hasattr(axes, "__len__") else axes
        ax.axis("off")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# PlotDendrogram - Gene dendrogram with module colors
# ======================================================================== #


def plot_dendrogram(
    adata: AnnData,
    group_labels: str = "Module Colors",
    hang: float = 0.03,
    add_guide: bool = True,
    guide_hang: float = 0.05,
    main: str = "",
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot WGCNA gene dendrogram with module color bars.

    Replicates R's PlotDendrogram / plotDendroAndColors function.
    The color bar below shows each gene's module assignment aligned to dendrogram leaves.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    hang : float
        Leaf hanging proportion
    add_guide : bool
        Add guide lines
    main : str
        Main title
    wgcna_name : str
        Experiment name
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)

    Z = wd.get("linkage_matrix")
    if Z is None:
        Z = wd.get("linkage")
    if Z is None:
        tom_dissim = wd.get("tom_dissim")
        if tom_dissim is not None:
            from scipy.spatial.distance import squareform
            from scipy.cluster.hierarchy import linkage as scipy_linkage

            print("  No linkage_matrix found, computing from TOM dissimilarity...")
            tom_dissim_arr = np.asarray(tom_dissim)
            tom_dissim_sym = (tom_dissim_arr + tom_dissim_arr.T) / 2.0
            np.fill_diagonal(tom_dissim_sym, 0)
            tom_dissim_sym = np.clip(tom_dissim_sym, 0, 2)
            condensed = squareform(tom_dissim_sym, checks=False)
            Z = scipy_linkage(condensed, method="average")
            wn = _get_wgcna_name(adata, wgcna_name)
            adata.uns["hdWGCNA"][wn]["linkage_matrix"] = Z
            wd = adata.uns["hdWGCNA"][wn]
            print(f"  Linkage matrix computed: shape={Z.shape}")
        else:
            raise ValueError(
                "No linkage matrix or TOM dissimilarity found. Run construct_network() first."
            )

    modules_df = wd.get("modules_df")

    if modules_df is None or len(modules_df) == 0:
        raise ValueError("No module assignments found.")

    gene_names = wd.get("dat_expr_genes", None)
    if gene_names is None:
        gene_names = (
            modules_df["gene_name"].tolist()
            if "gene_name" in modules_df.columns
            else list(range(len(modules_df)))
        )

    gene_to_color = dict(zip(modules_df["gene_name"], modules_df["color"]))

    n_genes = Z.shape[0] + 1
    gene_order = leaves_list(Z)

    max_height = Z[:, 2].max()
    hang_level = hang * max_height

    x_pos = np.zeros(2 * n_genes)
    y_pos = np.zeros(2 * n_genes)

    for pos, leaf_idx in enumerate(gene_order):
        x_pos[int(leaf_idx)] = float(pos)
        y_pos[int(leaf_idx)] = 0.0

    parent_merge_h = np.zeros(n_genes)

    for i in range(Z.shape[0]):
        left_id = int(Z[i, 0])
        right_id = int(Z[i, 1])
        merge_h = float(Z[i, 2])
        new_id = n_genes + i

        x_pos[new_id] = (x_pos[left_id] + x_pos[right_id]) / 2.0
        y_pos[new_id] = merge_h

        for child_id in [left_id, right_id]:
            if child_id < n_genes:
                parent_merge_h[child_id] = merge_h

    for i in range(n_genes):
        hang_y = max(0.0, parent_merge_h[i] - hang_level)
        y_pos[i] = hang_y

    fig = plt.figure(figsize=(8, 3.5), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[1, 0.08], hspace=0.02)

    ax_dendro = fig.add_subplot(gs[0])
    ax_color = fig.add_subplot(gs[1])

    for i in range(Z.shape[0]):
        left_id = int(Z[i, 0])
        right_id = int(Z[i, 1])
        merge_h = float(Z[i, 2])

        left_x_val = x_pos[left_id]
        right_x_val = x_pos[right_id]
        left_y = y_pos[left_id]
        right_y = y_pos[right_id]

        ax_dendro.plot(
            [left_x_val, left_x_val], [left_y, merge_h], color="#555555", linewidth=0.6
        )
        ax_dendro.plot(
            [right_x_val, right_x_val],
            [right_y, merge_h],
            color="#555555",
            linewidth=0.6,
        )
        ax_dendro.plot(
            [left_x_val, right_x_val],
            [merge_h, merge_h],
            color="#555555",
            linewidth=0.6,
        )

    min_leaf_y = min(y_pos[i] for i in range(n_genes))

    ax_dendro.set_xlim(-1, n_genes)
    ax_dendro.set_ylim(min_leaf_y - hang_level * 0.5, 1.0)
    ax_dendro.set_xticks([])
    y_ticks = np.linspace(min_leaf_y, 1.0, num=5)
    y_ticks = np.round(y_ticks, 2)
    ax_dendro.set_yticks(y_ticks)
    ax_dendro.set_yticklabels([f"{v:.2f}" for v in y_ticks], fontsize=11)
    ax_dendro.tick_params(axis="y", pad=2)
    ax_dendro.set_ylabel("Height", fontsize=12, rotation=90, labelpad=2)
    ax_dendro.spines["bottom"].set_visible(False)
    ax_dendro.spines["top"].set_visible(False)
    ax_dendro.spines["right"].set_visible(False)
    ax_dendro.spines["left"].set_linewidth(0.8)
    ax_dendro.spines["left"].set_color("#333333")
    ax_dendro.spines["left"].set_position(("outward", 20))
    if main:
        ax_dendro.set_title(main, fontsize=14, fontweight="bold")

    color_ordered = []
    for leaf_idx in gene_order:
        leaf_idx = int(leaf_idx)
        if leaf_idx < len(gene_names):
            gname = gene_names[leaf_idx]
            color_ordered.append(gene_to_color.get(gname, "grey"))
        else:
            color_ordered.append("grey")

    blocks = []
    cur_color = color_ordered[0]
    block_start = 0
    for i in range(1, len(color_ordered)):
        if color_ordered[i] != cur_color:
            blocks.append((block_start, i - block_start, cur_color))
            cur_color = color_ordered[i]
            block_start = i
    blocks.append((block_start, len(color_ordered) - block_start, cur_color))

    for start, width, color in blocks:
        ax_color.bar(start, 1, width=width, color=color, align="edge")
    ax_color.set_xlim(-1, n_genes)
    ax_color.set_ylim(0, 1)
    ax_color.set_xticks([])
    ax_color.set_yticks([])
    ax_color.set_ylabel(
        group_labels, fontsize=11, rotation=0, ha="right", va="center", labelpad=3
    )
    ax_color.spines["left"].set_linewidth(0.8)
    ax_color.spines["left"].set_color("#333333")
    ax_color.spines["bottom"].set_visible(False)
    ax_color.spines["top"].set_visible(False)
    ax_color.spines["right"].set_visible(False)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# PlotKMEs - kME barplots per module
# ======================================================================== #


def plot_kmes(
    adata: AnnData,
    n_hubs: int = 10,
    text_size: int = 6,
    ncols: int = 4,
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot genes ranked by kME (eigengene-based connectivity) per module.

    Replicates R's PlotKMEs function.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    n_hubs : int
        Number of top hub genes to show
    text_size : int
        Font size for gene labels
    ncols : int
        Number of columns
    wgcna_name : str
        Experiment name
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")

    if modules_df is None:
        raise ValueError("No module data found. Run construct_network() first.")

    kme_col = None
    if "kME" in modules_df.columns:
        kme_col = "kME"
    else:
        kme_cols = [
            c
            for c in modules_df.columns
            if c.startswith("kME_") or c.upper().startswith("KME")
        ]
        if len(kme_cols) > 0:
            kme_col = kme_cols[0]
        else:
            kme_cols = [c for c in modules_df.columns if "kME" in c.lower()]
            if len(kme_cols) > 0:
                kme_col = kme_cols[0]

    if kme_col is None:
        print(f"  Available columns: {list(modules_df.columns)}")
        raise ValueError("No kME values found. Run module_connectivity() first.")

    mods_df = modules_df[modules_df["module"] != "grey"].copy()
    if len(mods_df) == 0:
        mods_df = modules_df.copy()

    unique_mods = sorted(mods_df["module"].unique())
    mod_colors = dict(zip(mods_df["module"], mods_df["color"]))

    n_mods = len(unique_mods)
    actual_ncols = min(ncols, n_mods)
    nrows = (n_mods + actual_ncols - 1) // actual_ncols

    panel_w = 2.5
    panel_h = 2.2
    fig, axes = plt.subplots(
        nrows,
        actual_ncols,
        figsize=(actual_ncols * panel_w, nrows * panel_h),
        constrained_layout=True,
    )

    if nrows == 1 and actual_ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif actual_ncols == 1:
        axes = axes.reshape(-1, 1)

    for idx, cur_mod in enumerate(unique_mods):
        row, col = idx // actual_ncols, idx % actual_ncols
        ax = axes[row, col]

        mod_genes = mods_df[mods_df["module"] == cur_mod].copy()
        mod_genes = mod_genes.sort_values(kme_col, ascending=True)

        top_n = min(n_hubs, len(mod_genes))
        display_genes = mod_genes.tail(top_n)

        cur_color = mod_colors.get(cur_mod, "#0072B2")

        y_pos = np.arange(len(display_genes))
        ax.barh(
            y_pos,
            display_genes[kme_col].values,
            color=cur_color,
            edgecolor=cur_color,
            height=0.8,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            display_genes["gene_name"].values
            if "gene_name" in display_genes.columns
            else display_genes.index,
            fontsize=text_size,
            style="italic",
        )
        ax.set_title(cur_mod, fontsize=11, fontweight="bold")
        ax.set_xlabel("kME", fontsize=10)
        ax.tick_params(axis="y", which="both", left=False, right=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    total_cells = nrows * actual_ncols
    for idx in range(len(unique_mods), total_cells):
        row, col = idx // actual_ncols, idx % actual_ncols
        try:
            ax = axes[row, col] if hasattr(axes, "__len__") else axes
            ax.axis("off")
        except Exception:
            pass

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# ModuleCorrelogram - Module eigengene correlation heatmap
# ======================================================================== #


def module_correlogram(
    adata: AnnData,
    features: str = "hMEs",
    exclude_grey: bool = True,
    method: str = "ellipse",
    type: str = "upper",
    order: str = "original",
    sig_level: float = 0.05,
    pch_cex: float = 1.5,
    tl_srt: float = 45,
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot module eigengene correlogram.

    Replicates R's ModuleCorrelogram using corrplot-style visualization.
    Uses ellipse method with upper triangle display, matching R's
    seagreen-white-darkorchid1 color scheme and significance marking.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    features : str
        'hMEs', 'MEs', 'scores', or 'average'
    exclude_grey : bool
        Exclude grey module
    method : str
        'ellipse' or 'color' or 'number'
    type : str
        'upper', 'lower', or 'full'
    order : str
        'original', 'AOE', 'FPC', 'hclust', or 'alphabet'
    sig_level : float
        Significance level for p-value marking
    pch_cex : float
        Size of significance markers
    tl_srt : float
        Text label rotation angle
    wgcna_name : str
        Experiment name
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)

    me_key = {
        "hMEs": "hMEs",
        "MEs": "MEs",
        "scores": "module_scores",
        "average": "avg_module_expr",
    }.get(features, "hMEs")
    MEs = wd.get(me_key)

    if MEs is None:
        raise ValueError(f"No {features} found.")

    if isinstance(MEs, np.ndarray):
        mod_names = wd.get("module_names", [f"M{i}" for i in range(MEs.shape[1])])
        MEs = pd.DataFrame(MEs, columns=mod_names)

    if exclude_grey and "grey" in MEs.columns:
        MEs = MEs.drop(columns=["grey"])

    from scipy import stats

    n_mods = MEs.shape[1]
    mod_names = list(MEs.columns)
    cor_mat = MEs.corr(method="pearson")
    p_mat = pd.DataFrame(np.ones((n_mods, n_mods)), index=mod_names, columns=mod_names)
    for i in range(n_mods):
        for j in range(i + 1, n_mods):
            r_val, p_val = stats.pearsonr(MEs.iloc[:, i].values, MEs.iloc[:, j].values)
            p_mat.iloc[i, j] = p_val
            p_mat.iloc[j, i] = p_val

    if order == "hclust":
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        dist_mat = 1.0 - cor_mat.values
        np.fill_diagonal(dist_mat, 0)
        dist_mat = np.maximum(dist_mat, 0)
        try:
            Z = linkage(squareform(dist_mat), method="average")
            leaf_order = leaves_list(Z)
        except Exception:
            leaf_order = list(range(n_mods))
        cor_mat = cor_mat.iloc[leaf_order, leaf_order]
        p_mat = p_mat.iloc[leaf_order, leaf_order]
        mod_names = [mod_names[i] for i in leaf_order]
    elif order == "alphabet":
        sorted_idx = sorted(range(n_mods), key=lambda x: mod_names[x])
        cor_mat = cor_mat.iloc[sorted_idx, sorted_idx]
        p_mat = p_mat.iloc[sorted_idx, sorted_idx]
        mod_names = [mod_names[i] for i in sorted_idx]

    from matplotlib.colors import LinearSegmentedColormap

    r_colors = ["#8FD3B0", "#FFFFFF", "#D050F0"]
    r_cmap = LinearSegmentedColormap.from_list(
        "seagreen_white_darkorchid", r_colors, N=256
    )

    clean_names = [str(n).replace("OPC-", "") for n in mod_names]

    n = len(mod_names)
    cell_size = max(0.6, 1.2 - n * 0.05)
    fig_size = max(3, n * cell_size + 1.5)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    from matplotlib.patches import Ellipse

    for i in range(n):
        for j in range(n):
            if type == "upper" and j < i:
                continue
            if type == "lower" and j > i:
                continue
            r_val = cor_mat.iloc[i, j]
            cx, cy = j, i

            if method == "ellipse":
                rect = plt.Rectangle(
                    (cx - 0.5, cy - 0.5),
                    1,
                    1,
                    facecolor="white",
                    edgecolor="#CCCCCC",
                    linewidth=0.5,
                    zorder=1,
                )
                ax.add_patch(rect)
                w = 0.85 * (1.0 - abs(r_val)) + 0.15
                h = 0.85
                angle = 45 if r_val > 0 else -45
                ellipse = Ellipse(
                    (cx, cy),
                    width=w,
                    height=h,
                    angle=angle,
                    facecolor=r_cmap(0.5 + r_val / 2),
                    edgecolor="none",
                    zorder=2,
                )
                ax.add_patch(ellipse)
            elif method == "color":
                rect = plt.Rectangle(
                    (cx - 0.5, cy - 0.5),
                    1,
                    1,
                    facecolor=r_cmap(0.5 + r_val / 2),
                    edgecolor="#CCCCCC",
                    linewidth=0.5,
                    zorder=1,
                )
                ax.add_patch(rect)
            elif method == "number":
                rect = plt.Rectangle(
                    (cx - 0.5, cy - 0.5),
                    1,
                    1,
                    facecolor=r_cmap(0.5 + r_val / 2),
                    edgecolor="#CCCCCC",
                    linewidth=0.5,
                    zorder=1,
                )
                ax.add_patch(rect)
                txt_color = "white" if abs(r_val) > 0.6 else "black"
                ax.text(
                    cx,
                    cy,
                    f"{r_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=max(6, 10 - n // 3),
                    color=txt_color,
                    zorder=4,
                )

    for i in range(n):
        ax.text(
            i,
            -0.6,
            clean_names[i],
            ha="center",
            va="bottom",
            fontsize=max(7, 10 - n // 3),
            rotation=tl_srt,
            zorder=6,
        )
        ax.text(
            i - 0.55,
            i,
            clean_names[i],
            ha="right",
            va="center",
            fontsize=max(7, 10 - n // 3),
            zorder=6,
        )

    ax.set_xlim(-1.2, n - 0.2)
    ax.set_ylim(n - 0.2, -1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=r_cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Pearson r", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(pad=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# ModuleNetworkPlot - Circular network plot for each module
# ======================================================================== #


def module_network_plot(
    adata: AnnData,
    n_inner: int = 10,
    n_outer: int = 15,
    n_conns: int = 500,
    mods: Union[str, List[str]] = "all",
    outdir: str = None,
    wgcna_name: str = None,
    plot_size: Tuple[float, float] = (6, 6),
    edge_alpha: float = 0.25,
    edge_width: float = 1,
    vertex_label_cex: float = 1,
    vertex_size: float = 6,
) -> Dict[str, plt.Figure]:
    """
    Visualize top hub genes as circular network plots for each module.

    Replicates R's ModuleNetworkPlot function.
    Returns a dict mapping module name -> Figure. If outdir is provided,
    also saves PDFs to that directory.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results and TOM
    n_inner : int
        Genes on inner circle (hub genes)
    n_outer : int
        Genes on outer circle
    n_conns : int
        Top connections to show
    mods : str or list
        Modules to plot ("all" or list of module names)
    outdir : str or None
        Output directory for PDF files (optional, if None only returns figures)
    plot_size : tuple
        Figure size (width, height) in inches
    edge_alpha : float
        Edge transparency (0-1)
    edge_width : float
        Line width of edges
    vertex_label_cex : float
        Font scale for gene labels
    vertex_size : float
        Node size for igraph/networkx plot
    wgcna_name : str
        Experiment name

    Returns
    -------
    dict
        Mapping of module name -> matplotlib Figure
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx required. Install with: pip install networkx")
        return

    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")
    TOM_arr, tom_genes = _get_tom_similarity(wd)

    if TOM_arr is None:
        raise ValueError("No TOM found. Run construct_network() first.")
    if modules_df is None:
        raise ValueError("No module data found.")

    if mods == "all":
        plot_mods = sorted(set(modules_df["module"].unique()) - {"grey"})
    else:
        plot_mods = mods

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        print(f"Writing output files to {outdir}")

    fig_dict = {}
    n_hubs_total = n_inner + n_outer
    hub_df = modules_df[modules_df["module"].isin(plot_mods)].copy()
    kme_cols = [c for c in hub_df.columns if "kME" in c.lower()]
    if kme_cols:
        hub_df = hub_df.sort_values(["module", kme_cols[0]], ascending=[True, False])
        hub_df = hub_df.groupby("module").head(n_hubs_total)

    gene_to_tom_idx = {g: i for i, g in enumerate(tom_genes)}

    for cur_mod in plot_mods:
        cur_color = mod_color_lookup(modules_df, cur_mod)
        mod_hub_df = hub_df[hub_df["module"] == cur_mod]
        cur_genes = (
            mod_hub_df["gene_name"].values
            if "gene_name" in mod_hub_df.columns
            else mod_hub_df.index.values
        )
        n_genes = len(cur_genes)

        if n_genes < (n_inner + 1):
            print(f"Skipping {cur_mod}, too few genes ({n_genes}).")
            continue

        print(f"Processing {cur_mod} ({n_genes} genes)...")

        valid_indices = [gene_to_tom_idx[g] for g in cur_genes if g in gene_to_tom_idx]
        if len(valid_indices) < 2:
            continue
        valid_indices = np.array(valid_indices)
        reduced_TOM = TOM_arr[np.ix_(valid_indices, valid_indices)]

        flat_vals = reduced_TOM.ravel()
        flat_vals[flat_vals < 0] = 0
        order_idx = np.argsort(flat_vals)[::-1]
        n_possible = len(order_idx)
        cur_n_conns = min(n_conns, n_possible)

        keep_idx = order_idx[:cur_n_conns]
        mask = np.zeros_like(reduced_TOM, dtype=bool)
        mask.ravel()[keep_idx] = True
        reduced_TOM[~mask] = 0

        tom_max = reduced_TOM.max() if reduced_TOM.max() > 0 else 1
        reduced_TOM = reduced_TOM / tom_max

        np.fill_diagonal(reduced_TOM, 0)

        G = nx.from_numpy_array(reduced_TOM)
        G.remove_edges_from(nx.selfloop_edges(G))
        gene_labels = {i: str(cur_genes[i]) for i in range(n_genes)}

        pos = {}
        for i in range(n_inner):
            angle = 2 * np.pi * i / max(n_inner, 1)
            pos[i] = (0.5 * np.cos(angle), 0.5 * np.sin(angle))
        for i in range(n_inner, n_genes):
            n_outer_actual = max(n_genes - n_inner, 1)
            angle = 2 * np.pi * (i - n_inner) / n_outer_actual
            pos[i] = (1.0 * np.cos(angle), 1.0 * np.sin(angle))

        jitter_amount = 0.005
        rng = np.random.default_rng(42)
        for k in pos:
            pos[k] = (
                pos[k][0] + rng.uniform(-jitter_amount, jitter_amount),
                pos[k][1] + rng.uniform(-jitter_amount, jitter_amount),
            )

        fig, ax = plt.subplots(figsize=plot_size)

        edge_colors = [to_rgba(cur_color, alpha=edge_alpha) for _ in G.edges()]
        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            width=edge_width,
            alpha=edge_alpha,
            edge_color=edge_colors,
            connectionstyle="arc3,rad=0",
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            ax=ax,
            node_color=cur_color,
            node_size=vertex_size * 20,
            edgecolors="black",
            linewidths=0.5,
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels=gene_labels,
            ax=ax,
            font_size=vertex_label_cex * 6,
            font_family="sans-serif",
        )

        ax.axis("off")
        plt.tight_layout(pad=0.3)

        mod_name_clean = str(cur_mod).replace("OPC-", "")
        fig_dict[mod_name_clean] = fig

        if outdir is not None:
            save_file = os.path.join(outdir, f"{mod_name_clean}.pdf")
            fig.savefig(save_file, bbox_inches="tight", pad_inches=0.02, dpi=300)
            print(f"  Saved: {save_file}")

    return fig_dict


# ======================================================================== #
# HubGeneNetworkPlot - Combined hub gene network
# ======================================================================== #


def hub_gene_network_plot(
    adata: AnnData,
    mods: Union[str, List[str]] = "all",
    n_hubs: int = 6,
    n_other: int = 3,
    sample_edges: bool = True,
    edge_prop: float = 0.5,
    return_graph: bool = False,
    edge_alpha: float = 0.25,
    vertex_label_cex: float = 0.5,
    hub_vertex_size: float = 4,
    other_vertex_size: float = 1,
    wgcna_name: str = None,
    save_path: str = None,
):
    """
    Combined network plot with hub genes from multiple modules.

    Replicates R's HubGeneNetworkPlot function exactly.
    Uses force-directed layout with module-colored nodes.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results and TOM
    mods : str or list
        Modules to include ("all" or list)
    n_hubs : int
        Hub genes per module
    n_other : int
        Non-hub genes per module (randomly sampled)
    sample_edges : bool
        Randomly sample edges (True) or take strongest (False)
    edge_prop : float
        Proportion of edges to keep
    return_graph : bool
        Return igraph/networkx object instead of plotting
    edge_alpha : float
        Edge transparency
    vertex_label_cex : float
        Label font size
    hub_vertex_size : float
        Size of hub gene nodes
    other_vertex_size : float
        Size of non-hub nodes
    wgcna_name : str
        Experiment name
    save_path : str
        Save path for figure

    Returns
    -------
    Figure or graph object
    """
    try:
        import networkx as nx
    except ImportError:
        print("networkx required. Install with: pip install networkx")
        return None

    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")
    MEs = wd.get("hMEs")
    if MEs is None:
        MEs = wd.get("MEs")
    TOM_arr, tom_genes = _get_tom_similarity(wd)

    if TOM_arr is None:
        raise ValueError("No TOM found.")
    if modules_df is None:
        raise ValueError("No module data found.")

    if mods == "all":
        plot_mods = sorted(set(modules_df["module"].unique()) - {"grey"})
    else:
        plot_mods = list(mods)

    gene_to_tom_idx = {g: i for i, g in enumerate(tom_genes)}

    hub_list = {}
    for cur_mod in plot_mods:
        mod_df = modules_df[modules_df["module"] == cur_mod].copy()
        kme_col = f"kME_{cur_mod}"
        if kme_col in mod_df.columns:
            mod_df = mod_df.nlargest(n_hubs, columns=kme_col)
        else:
            kme_cols = [c for c in mod_df.columns if "kME" in c.lower()]
            if kme_cols:
                mod_df = mod_df.nlargest(n_hubs, columns=kme_cols[0])
            else:
                mod_df = mod_df.head(n_hubs)
        hub_genes = (
            mod_df["gene_name"].values
            if "gene_name" in mod_df.columns
            else mod_df.index.values
        )
        hub_list[cur_mod] = list(hub_genes)[:n_hubs]

    all_hub_genes = []
    for v in hub_list.values():
        all_hub_genes.extend(v)
    all_hub_genes = list(dict.fromkeys(all_hub_genes))

    other_genes = []
    non_hub = modules_df[~modules_df["gene_name"].isin(all_hub_genes)]
    rng_other = np.random.default_rng(42)
    for cur_mod in plot_mods:
        mod_non_hub = non_hub[non_hub["module"] == cur_mod]
        if len(mod_non_hub) > 0:
            n_sample = min(n_other, len(mod_non_hub))
            sampled = mod_non_hub.sample(
                n=n_sample, replace=True, random_state=rng_other.integers(0, 2**31)
            )
            other_genes.extend(sampled["gene_name"].tolist())
    other_genes = list(dict.fromkeys(other_genes))

    selected_genes = all_hub_genes + other_genes

    selected_modules = modules_df[modules_df["gene_name"].isin(selected_genes)].copy()
    gene_to_mod_row = {}
    for _, row in selected_modules.iterrows():
        gn = row["gene_name"] if "gene_name" in selected_modules.columns else row.name
        gene_to_mod_row[gn] = row.to_dict()

    selected_modules["geneset"] = selected_modules["gene_name"].apply(
        lambda x: "other" if x in set(other_genes) else "hub"
    )
    selected_modules["size"] = selected_modules["geneset"].apply(
        lambda x: hub_vertex_size if x == "hub" else other_vertex_size
    )
    selected_modules["label"] = selected_modules.apply(
        lambda x: str(x["gene_name"]) if x["geneset"] == "hub" else "", axis=1
    )

    valid_genes = [g for g in selected_genes if g in gene_to_tom_idx]
    if len(valid_genes) < 2:
        print("Too few valid genes for network plot.")
        return None

    valid_tom_idx = np.array([gene_to_tom_idx[g] for g in valid_genes])
    subset_TOM = TOM_arr[np.ix_(valid_tom_idx, valid_tom_idx)]
    subset_TOM = np.maximum(subset_TOM, 0)

    _gene_to_node_idx = {g: i for i, g in enumerate(valid_genes)}  # noqa: F841

    n_valid = len(valid_genes)

    mod_gene_indices = {}
    for g in valid_genes:
        mr = gene_to_mod_row.get(g)
        if isinstance(mr, dict):
            mc = _to_mpl_color(mr.get("color", "grey"))
            mod_gene_indices.setdefault(mc, []).append(valid_genes.index(g))

    edge_data = []

    for mc, indices in mod_gene_indices.items():
        n_genes_in_mod = len(indices)
        if n_genes_in_mod < 2:
            continue
        pair_vals = []
        for ii in range(n_genes_in_mod):
            for jj in range(ii + 1, n_genes_in_mod):
                vv = subset_TOM[indices[ii], indices[jj]]
                pair_vals.append((vv, indices[ii], indices[jj]))
        pair_vals.sort(key=lambda x: -x[0])
        n_intra_keep = min(
            len(pair_vals),
            max(3 * n_genes_in_mod, n_genes_in_mod * (n_genes_in_mod - 1) // 2),
        )
        for vv, ii, jj in pair_vals[:n_intra_keep]:
            edge_data.append(
                {
                    "from": valid_genes[ii],
                    "to": valid_genes[jj],
                    "value": vv,
                    "color_raw": mc,
                }
            )

    inter_pair_vals = []
    for i in range(n_valid):
        for j in range(i + 1, n_valid):
            g1, g2 = valid_genes[i], valid_genes[j]
            r1 = gene_to_mod_row.get(g1)
            r2 = gene_to_mod_row.get(g2)
            c1 = (
                _to_mpl_color(r1.get("color", "grey"))
                if isinstance(r1, dict)
                else "grey"
            )
            c2 = (
                _to_mpl_color(r2.get("color", "grey"))
                if isinstance(r2, dict)
                else "grey"
            )
            if c1 != c2:
                inter_pair_vals.append((subset_TOM[i, j], i, j))

    inter_pair_vals.sort(key=lambda x: -x[0])
    n_inter_keep = min(
        len(inter_pair_vals), max(3 * len(mod_gene_indices), len(mod_gene_indices) + 5)
    )
    for vv, i, j in inter_pair_vals[:n_inter_keep]:
        edge_data.append(
            {
                "from": valid_genes[i],
                "to": valid_genes[j],
                "value": vv,
                "color_raw": "#E6E6E6",
            }
        )

    n_intra = sum(1 for e in edge_data if e["color_raw"] != "#E6E6E6")
    n_inter = sum(1 for e in edge_data if e["color_raw"] == "#E6E6E6")
    print(f"  Edges: {n_intra} intra-module, {n_inter} inter-module")

    if len(edge_data) == 0:
        print("No edges found.")
        return None

    edge_df = pd.DataFrame(edge_data)

    groups = edge_df["color_raw"].unique()

    sampled_parts = []
    for grp in groups:
        grp_df = edge_df[edge_df["color_raw"] == grp]
        n_edges = len(grp_df)
        n_keep = max(1, round(n_edges * edge_prop))
        if sample_edges:
            rng_s = np.random.default_rng(42)
            keep_idx = rng_s.choice(n_edges, size=n_keep, replace=False)
            sampled_parts.append(grp_df.iloc[keep_idx])
        else:
            sampled_parts.append(grp_df.nlargest(n_keep, columns="value"))

    edge_df = pd.concat(sampled_parts, ignore_index=True)

    def scale01(x):
        xmin = x.min()
        xmax = x.max()
        if xmax == xmin:
            return pd.Series([1.0] * len(x), index=x.index)
        return (x - xmin) / (xmax - xmin)

    edge_df["value_scaled"] = edge_df.groupby("color_raw")["value"].transform(scale01)

    edge_df["color_final"] = edge_df.apply(
        lambda row: to_rgba(
            _to_mpl_color(row["color_raw"]), alpha=float(row["value_scaled"])
        ),
        axis=1,
    )

    G = nx.Graph()
    for g in valid_genes:
        mr = gene_to_mod_row.get(g)
        if not isinstance(mr, dict):
            continue
        is_hub = g not in set(other_genes)
        G.add_node(
            g,
            color=_to_mpl_color(mr.get("color", "grey")),
            size=hub_vertex_size if is_hub else other_vertex_size,
            label=str(g) if is_hub else "",
            geneset="hub" if is_hub else "other",
        )

    for _, row in edge_df.iterrows():
        G.add_edge(
            row["from"],
            row["to"],
            weight=row["value"],
            color=row["color_final"],
            color_raw=row["color_raw"],
            value_scaled=row["value_scaled"],
        )

    n_nodes = G.number_of_nodes()
    if n_nodes < 2:
        print("Too few nodes for layout.")
        return None

    try:
        import igraph as ig

        node_names = list(G.nodes())
        node_idx = {name: i for i, name in enumerate(node_names)}
        ig_edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]

        layout_weights = [float(G[u][v]["value_scaled"]) for u, v in G.edges()]
        layout_weights = [max(0.001, w) for w in layout_weights]

        ig_g = ig.Graph(n=n_nodes, edges=ig_edges, directed=False)
        print(f"  igraph FR layout: {n_nodes} nodes, {len(ig_edges)} edges")
        print(f"  weight range: [{min(layout_weights):.4f}, {max(layout_weights):.4f}]")
        layout = ig_g.layout_fruchterman_reingold(weights=layout_weights)
        pos = {node_names[i]: (layout[i][0], layout[i][1]) for i in range(n_nodes)}
    except Exception as e:
        print(f"  igraph layout failed ({e}), falling back to networkx")
        pos = nx.spring_layout(
            G, seed=42, k=1.5 / np.sqrt(max(n_nodes, 1)), iterations=500
        )

    if return_graph:
        return G

    fig, ax = plt.subplots(figsize=(10, 10))

    for u, v in G.edges():
        ed = G[u][v]
        color = ed["color"]
        alpha_val = float(ed["value_scaled"]) * edge_alpha
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=color,
            alpha=alpha_val,
            linewidth=0.5,
            zorder=1,
        )

    node_colors = [G.nodes[n]["color"] for n in G.nodes()]
    node_sizes = [G.nodes[n]["size"] * 40 for n in G.nodes()]
    labels = {n: G.nodes[n]["label"] for n in G.nodes() if G.nodes[n]["label"]}

    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=node_colors,
        linewidths=0.5,
    )
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax, font_size=vertex_label_cex * 8, font_color="black"
    )

    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# ModuleUMAPPlot - Gene UMAP with TOM edges
# ======================================================================== #


def module_umap_plot(
    adata: AnnData,
    sample_edges: bool = True,
    edge_prop: float = 0.2,
    label_hubs: int = 5,
    edge_alpha: float = 0.25,
    vertex_label_cex: float = 0.5,
    label_genes: List[str] = None,
    return_graph: bool = False,
    keep_grey_edges: bool = True,
    wgcna_name: str = None,
    save_path: str = None,
):
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")
    TOM_arr, tom_genes = _get_tom_similarity(wd)
    umap_df = wd.get("module_umap")

    if TOM_arr is None:
        raise ValueError("No TOM found.")
    if umap_df is None:
        raise ValueError("No module UMAP found. Run compute_module_umap() first.")

    mods = sorted(set(modules_df["module"].unique()) - {"grey"})

    gene_to_tom_idx = {g: i for i, g in enumerate(tom_genes)}

    hub_list = {}
    for cur_mod in mods:
        mod_df = modules_df[modules_df["module"] == cur_mod].copy()
        kme_col = f"kME_{cur_mod}"
        if kme_col in mod_df.columns:
            mod_df = mod_df.nlargest(label_hubs, columns=kme_col)
        else:
            alt_kme = [c for c in mod_df.columns if "kME" in c.lower()]
            if alt_kme:
                mod_df = mod_df.nlargest(label_hubs, columns=alt_kme[0])
        hub_list[cur_mod] = mod_df["gene_name"].values[:label_hubs].tolist()

    hub_labels = []
    for v in hub_list.values():
        hub_labels.extend(v)
    hub_labels = list(dict.fromkeys(hub_labels))

    if label_genes is not None:
        hub_labels = list(set(hub_labels) | set(label_genes))

    umap_genes = (
        umap_df["gene"].values if "gene" in umap_df.columns else umap_df.index.values
    )
    _n_genes = len(umap_genes)  # noqa: F841

    gene_name_to_color = dict(zip(modules_df["gene_name"], modules_df["color"]))
    gene_name_to_module = dict(zip(modules_df["gene_name"], modules_df["module"]))

    gene_name_to_kme = {}
    kme_cols = [c for c in modules_df.columns if "kME" in c.lower()]
    for _, row in modules_df.iterrows():
        gn = row["gene_name"]
        mod = row.get("module", "")
        kme_val = 0.0
        mod_kme = f"kME_{mod}"
        if mod_kme in row.index:
            kme_val = float(row[mod_kme])
        elif kme_cols:
            kme_val = float(row[kme_cols[0]])
        gene_name_to_kme[gn] = kme_val

    selected = pd.DataFrame(
        {
            "gene_name": umap_genes,
            "UMAP1": umap_df["UMAP1"].values
            if "UMAP1" in umap_df.columns
            else umap_df.values[:, 0],
            "UMAP2": umap_df["UMAP2"].values
            if "UMAP2" in umap_df.columns
            else umap_df.values[:, 1],
            "hub": umap_df["hub"].values if "hub" in umap_df.columns else "other",
            "kME": umap_df["kME"].values
            if "kME" in umap_df.columns
            else [gene_name_to_kme.get(g, 0) for g in umap_genes],
            "color": [gene_name_to_color.get(g, "grey") for g in umap_genes],
            "module": [gene_name_to_module.get(g, "grey") for g in umap_genes],
        }
    )
    selected["label"] = selected["gene_name"].apply(
        lambda x: x if x in hub_labels else ""
    )
    selected["fontcolor"] = selected["color"].apply(
        lambda c: "#808080" if c == "black" else "black"
    )
    selected["framecolor"] = selected.apply(
        lambda row: "black" if row["gene_name"] in hub_labels else row["color"], axis=1
    )

    hub_genes_in_umap = selected[selected["hub"] == "hub"]["gene_name"].tolist()

    gene_to_umap_idx = {g: i for i, g in enumerate(umap_genes)}

    hub_tom_idx = []
    hub_umap_idx = []
    for hg in hub_genes_in_umap:
        if hg in gene_to_tom_idx:
            hub_tom_idx.append(gene_to_tom_idx[hg])
            hub_umap_idx.append(gene_to_umap_idx[hg])

    if len(hub_tom_idx) < 2:
        print("Not enough hub genes for UMAP plot.")
        return None

    all_tom_idx = [gene_to_tom_idx[g] for g in umap_genes if g in gene_to_tom_idx]
    all_umap_idx = [gene_to_umap_idx[g] for g in umap_genes if g in gene_to_tom_idx]

    subset_TOM = TOM_arr[np.ix_(all_tom_idx, hub_tom_idx)]

    edge_data = []
    for i in range(len(all_tom_idx)):
        for j in range(len(hub_tom_idx)):
            ui = all_umap_idx[i]
            uj = hub_umap_idx[j]
            if ui == uj:
                continue
            val = subset_TOM[i, j]
            if val <= 0:
                continue
            g1 = umap_genes[ui]
            g2 = umap_genes[uj]
            c1 = gene_name_to_color.get(g1, "grey")
            c2 = gene_name_to_color.get(g2, "grey")
            ec = c1 if c1 == c2 else "grey90"
            edge_data.append(
                {
                    "src": ui,
                    "dst": uj,
                    "gene1": g1,
                    "gene2": g2,
                    "value": val,
                    "color_raw": ec,
                }
            )

    edge_df = pd.DataFrame(edge_data)
    if edge_df.empty:
        print("No edges found.")
        return None

    if not keep_grey_edges:
        edge_df = edge_df[edge_df["color_raw"] != "grey90"].copy()

    groups = edge_df["color_raw"].unique()
    temp_list = []
    rng = np.random.default_rng(42)
    for grp in groups:
        grp_df = edge_df[edge_df["color_raw"] == grp].copy()
        n_edges = len(grp_df)
        n_k = max(1, int(round(n_edges * edge_prop)))
        if sample_edges:
            sample_idx = rng.choice(n_edges, size=n_k, replace=False)
            temp_list.append(grp_df.iloc[sample_idx])
        else:
            grp_df = grp_df.nlargest(n_k, "value")
            temp_list.append(grp_df)
    edge_df = pd.concat(temp_list, ignore_index=True)

    def _scale01(x):
        xmin, xmax = x.min(), x.max()
        if xmax == xmin:
            return pd.Series(0.5, index=x.index)
        return (x - xmin) / (xmax - xmin)

    edge_df["value_scaled"] = edge_df.groupby("color_raw")["value"].transform(_scale01)

    edge_df = edge_df.sort_values("value_scaled")
    grey_mask = edge_df["color_raw"] == "grey90"
    edge_df = pd.concat([edge_df[grey_mask], edge_df[~grey_mask]], ignore_index=True)

    def _to_mpl_color(c):
        color_map = {
            "grey90": "#E6E6E6",
            "grey60": "#999999",
            "greenyellow": "#ADFF2F",
            "lightyellow": "#FFFFE0",
            "lightcyan": "#E0FFFF",
            "midnightblue": "#191970",
            "skyblue": "#87CEEB",
            "darkgrey": "#A9A9A9",
        }
        if c in color_map:
            return color_map[c]
        try:
            return mcolors.to_hex(c)
        except Exception:
            return c

    edge_df["color_final"] = edge_df.apply(
        lambda row: to_rgba(
            _to_mpl_color(row["color_raw"]),
            alpha=float(row["value_scaled"]) / 2
            if row["color_raw"] == "grey90"
            else float(row["value_scaled"]),
        ),
        axis=1,
    )

    other_mask = selected["hub"] == "other"
    no_label_mask = selected["label"] == ""
    selected = pd.concat(
        [
            selected[other_mask & no_label_mask],
            selected[other_mask & ~no_label_mask],
            selected[~other_mask & no_label_mask],
            selected[~other_mask & ~no_label_mask],
        ],
        ignore_index=True,
    )

    selected["orig_idx"] = selected["gene_name"].map(gene_to_umap_idx)

    if return_graph:
        try:
            import networkx as nx

            G = nx.Graph()
            for _, row in selected.iterrows():
                G.add_node(row["gene_name"], **row.to_dict())
            for _, row in edge_df.iterrows():
                G.add_edge(
                    row["gene1"],
                    row["gene2"],
                    weight=row["value"],
                    color=row["color_final"],
                )
            return G
        except ImportError:
            return None

    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgba as mpl_to_rgba

    fig, ax = plt.subplots(figsize=(8, 8))

    coords = selected[["UMAP1", "UMAP2"]].values

    edge_lines = []
    edge_colors_list = []
    gene_to_selected_idx = {row["gene_name"]: idx for idx, row in selected.iterrows()}
    for _, row in edge_df.iterrows():
        si = gene_to_selected_idx.get(row["gene1"])
        di = gene_to_selected_idx.get(row["gene2"])
        if si is None or di is None:
            continue
        p1 = [selected.loc[si, "UMAP1"], selected.loc[si, "UMAP2"]]
        p2 = [selected.loc[di, "UMAP1"], selected.loc[di, "UMAP2"]]
        edge_lines.append([p1, p2])
        edge_colors_list.append(row["color_final"])

    if edge_lines:
        lc = LineCollection(
            edge_lines, colors=edge_colors_list, linewidths=0.5, alpha=edge_alpha
        )
        ax.add_collection(lc)

    kME_vals = selected["kME"].values.astype(float)
    kME_vals = np.nan_to_num(kME_vals, nan=0.0)

    node_colors_raw = selected["color"].values
    node_framecolors = selected["framecolor"].values

    node_colors_rgba = []
    for i in range(len(selected)):
        base_hex = _to_mpl_color(node_colors_raw[i])
        base_rgba = mpl_to_rgba(base_hex)
        kme_norm = kME_vals[i]
        alpha_val = 0.3 + 0.7 * kme_norm
        node_colors_rgba.append((*base_rgba[:3], alpha_val))

    node_sizes = kME_vals * 150 + 5

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        s=node_sizes,
        c=node_colors_rgba,
        edgecolors=node_framecolors,
        linewidths=0.5,
        zorder=3,
    )

    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()

    label_rows = selected[selected["label"] != ""]
    if len(label_rows) > 0:
        texts = []
        for _, row in label_rows.iterrows():
            t = ax.text(
                row["UMAP1"],
                row["UMAP2"],
                row["label"],
                fontsize=vertex_label_cex * 8,
                fontstyle="italic",
                fontweight="bold",
                color=row["fontcolor"],
                ha="center",
                va="center",
                zorder=5,
            )
            texts.append(t)

        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(
                    arrowstyle="-",
                    color="#555555",
                    lw=0.5,
                    connectionstyle="arc3,rad=0.1",
                ),
                force_text=(0.3, 0.5),
                force_points=(0.2, 0.3),
                expand=(1.2, 1.4),
                avoid_self=True,
                only_move={"points": "xy", "text": "xy"},
                lim=int(1e3),
            )
        except ImportError:
            pass

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    x_pad = x_range * 0.08
    y_pad = y_range * 0.08
    ax.set_xlim(coords[:, 0].min() - x_pad, coords[:, 0].max() + x_pad)
    ax.set_ylim(coords[:, 1].min() - y_pad, coords[:, 1].max() + y_pad)
    ax.set_aspect("equal")

    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# DME Visualization Functions
# ======================================================================== #


def plot_dmes_volcano(
    adata: AnnData,
    dme_df: pd.DataFrame,
    plot_labels: bool = True,
    label_size: float = 4,
    mod_point_size: float = 4,
    show_cutoff: bool = True,
    wgcna_name: str = None,
    xlim_range: Tuple[float, float] = None,
    ylim_range: Tuple[float, float] = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Volcano plot for DME results.

    Replicates R's PlotDMEsVolcano function.

    Parameters
    ----------
    adata : AnnData
        AnnData object
    dme_df : DataFrame
        Output from find_dmes() or find_all_dmes()
    plot_labels : bool
        Show significant module labels
    label_size : float
        Label font size
    mod_point_size : float
        Point size
    show_cutoff : bool
        Show significance cutoff lines
    wgcna_name : str
        Experiment name
    xlim_range : tuple
        X-axis limits (log2FC)
    ylim_range : tuple
        Y-axis limits (-log10 p-value)
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")

    df = dme_df.copy().dropna(subset=["p_val", "avg_log2FC"])
    if len(df) == 0:
        print("No valid data for volcano plot.")
        return None

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["avg_log2FC"])

    lowest_nonzero = (
        df[df["p_val"] > 0]["p_val"].min() if any(df["p_val"] > 0) else 1e-10
    )
    df["p_val"] = df["p_val"].replace(0, lowest_nonzero)

    modules = modules_df[modules_df["module"].isin(df["module"])]
    mod_colors = dict(zip(modules["module"], modules["color"]))

    df["anno"] = df.apply(
        lambda r: r["module"] if r.get("p_val_adj", r["p_val"]) < 0.05 else "", axis=1
    )

    max_fc = np.max(np.abs(df["avg_log2FC"]))
    if xlim_range is None:
        xlim_range = (-max_fc - 0.1, max_fc + 0.1)

    ymax = np.max(-np.log10(df["p_val"]))
    if ylim_range is None:
        ylim_range = (0, ymax + 1)

    fig, ax = plt.subplots(figsize=(6, 5))

    if show_cutoff:
        ax.axvline(x=0, color="#BFBFBF", linestyle="--", alpha=0.8, linewidth=1)
        ax.axhspan(ymin=-np.inf, ymax=-np.log10(0.05), color="#BFBFBF", alpha=0.3)

    colors = [mod_colors.get(m, "#0072B2") for m in df["module"]]
    ax.scatter(
        df["avg_log2FC"],
        -np.log10(df["p_val"]),
        s=mod_point_size**2 * 5,
        c=colors,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
    )

    if plot_labels:
        sig = df[df["anno"] != ""]
        for _, row in sig.iterrows():
            ax.annotate(
                row["anno"],
                (row["avg_log2FC"], -np.log10(row["p_val"])),
                fontsize=label_size,
                color="black",
                ha="center",
                va="bottom",
            )

    ax.set_xlim(xlim_range)
    ax.set_ylim(ylim_range)
    ax.set_xlabel(r"Average $\log_{2}$(Fold Change)", fontsize=11)
    ax.set_ylabel(r"$-\log_{10}$(Adj. P-value)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.3)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved: {save_path}")

    return fig


def plot_dmes_lollipop(
    adata: AnnData,
    dme_df: pd.DataFrame,
    group_by: str = None,
    comparison: Union[str, List[str]] = None,
    pvalue: str = "p_val_adj",
    avg_log2fc_col: str = "avg_log2FC",
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Lollipop plot for DME results.

    Replicates R's PlotDMEsLollipop function.

    Parameters
    ----------
    adata : AnnData
        AnnData object
    dme_df : DataFrame
        DME results
    group_by : str
        Grouping column name
    comparison : str or list
        Specific comparisons to plot
    pvalue : str
        P-value column to use ('p_val' or 'p_val_adj')
    avg_log2fc_col : str
        LogFC column name
    wgcna_name : str
        Experiment name
    save_path : str
        Save path

    Returns
    -------
    Figure or list of Figures
    """
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")

    df = dme_df.copy().dropna(subset=[pvalue, avg_log2fc_col])
    if len(df) == 0:
        print("No valid data.")
        return None

    mods_avail = sorted(modules_df["module"].unique())
    mods_avail = [m for m in mods_avail if m != "grey"]
    mod_colors = {}
    for m in mods_avail:
        row = modules_df[modules_df["module"] == m]
        if len(row) > 0 and "color" in row.columns:
            mod_colors[m] = row["color"].values[0]

    comparisons = (
        df[group_by].unique()
        if group_by and group_by in df.columns
        else df.get("comparison", pd.Series(["default"])).unique()
    )

    if comparison is not None:
        if isinstance(comparison, str):
            comparisons = [comparison]
        comparisons = [c for c in comparisons if c in set(comparisons)]

    fig_list = {}
    for comp in comparisons:
        cdf = (
            df[df[group_by] == comp]
            if group_by and group_by in df.columns
            else df.copy()
        )
        cdf = cdf.sort_values(avg_log2fc_col, ascending=False)
        cdf["module"] = pd.Categorical(cdf["module"], categories=mods_avail)
        cdf = cdf.sort_values(avg_log2fc_col)

        n_genes_map = dict(zip(cdf["module"], cdf.groupby("module").size()))

        fig, ax = plt.subplots(figsize=(5, max(3, len(cdf) * 0.25)))

        colors = [mod_colors.get(m, "#0072B2") for m in cdf["module"]]
        ax.axvline(x=0, color="black", linewidth=0.8)

        y_pos = np.arange(len(cdf))
        for i, (_, row) in enumerate(cdf.iterrows()):
            fc = row[avg_log2fc_col]
            pv = row[pvalue]
            shape = "o" if pv < 0.05 else "x"
            n_g = n_genes_map.get(row["module"], 1)
            ax.scatter(
                fc,
                i,
                s=np.log(n_g + 1) * 30,
                c=mod_colors.get(row["module"], "#0072B2"),
                edgecolors="black",
                linewidths=0.5,
                zorder=5,
            )
            ax.plot(
                [0, fc], [i, i], color=colors[i], linewidth=0.8, alpha=0.4, zorder=3
            )
            ax.scatter(
                fc,
                i,
                marker=shape,
                s=40,
                c="white" if shape == "o" else "black",
                edgecolors="black",
                zorder=6,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cdf["module"].values, fontsize=9)
        ax.set_xlabel(r"Avg. $\log_{2}$(Fold Change)", fontsize=11)
        ax.set_title(str(comp), fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, right=False)

        plt.tight_layout(pad=0.3)

        if save_path:
            comp_safe = str(comp).replace("/", "_vs_")
            sp = save_path.replace(".pdf", f"_{comp_safe}.pdf")
            fig.savefig(sp, bbox_inches="tight", pad_inches=0.02, dpi=300)
            print(f"Saved: {sp}")

        fig_list[comp] = fig

    if len(fig_list) == 1:
        return list(fig_list.values())[0]
    return fig_list


# ======================================================================== #
# Module-Trait Correlation Plot
# ======================================================================== #


def plot_module_trait_correlation(
    adata: AnnData,
    high_color: str = "red",
    mid_color: str = "lightgrey",
    low_color: str = "blue",
    label: Optional[str] = None,
    label_symbol: str = "stars",
    plot_max: float = None,
    text_size: float = 2,
    text_color: str = "black",
    text_digits: int = 3,
    combine: bool = True,
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Heatmap of module-trait correlations.

    Replicates R's PlotModuleTraitCorrelation function.

    Parameters
    ----------
    adata : AnnData
        AnnData object
    high_color : str
        Color for positive correlation
    mid_color : str
        Color for zero correlation
    low_color : str
        Color for negative correlation
    label : str or None
        Show significance labels ('pval', 'fdr', or None)
    label_symbol : str
        'stars' or 'numeric'
    plot_max : float or None
        Max correlation value
    text_size : float
        Label text size
    text_color : str
        Label text color
    combine : bool
        Combine into single figure
    wgcna_name : str
        Experiment name
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    from .analysis import module_trait_correlation

    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)

    temp = module_trait_correlation(adata, trait_cols=[], wgcna_name=wgcna_name)
    if temp is None or len(temp["cor"]) == 0:
        print("No trait correlation data found.")
        return None

    modules_df = wd.get("modules_df")
    _mod_colors = modules_df.drop_duplicates("module")[["module", "color"]].set_index(
        "module"
    )

    plot_list = {}
    for tname, cor_mat in temp["cor"].items():
        pval_mat = temp["pval"].get(tname, None)
        fdr_mat = temp["fdr"].get(tname, None)

        cor_df = cor_mat.reset_index().melt(var_name="Module", value_name="cor")

        if label and pval_mat is not None:
            src_mat = fdr_mat if label == "fdr" else pval_mat
            p_df = src_mat.reset_index().melt(var_name="Module", value_name="pval")
            cor_df["pval"] = p_df["pval"]

            if label_symbol == "stars":

                def star_fn(p):
                    if p < 0.001:
                        return "***"
                    elif p < 0.01:
                        return "**"
                    elif p < 0.05:
                        return "*"
                    return ""

                cor_df["sig"] = cor_df["pval"].apply(star_fn)
            else:
                cor_df["sig"] = cor_df.apply(
                    lambda r: (
                        format(r["pval"], f".{text_digits}f")
                        if r["pval"] < 0.05
                        else ""
                    ),
                    axis=1,
                )

        if plot_max is None:
            plot_max = np.max(np.abs(cor_df["cor"].dropna()))
        cor_df["cor"] = cor_df["cor"].clip(lower=-plot_max, upper=plot_max)

        mods_plot = cor_df["Module"].unique()
        n_mods = len(mods_plot)

        fig, ax = plt.subplots(figsize=(max(4, n_mods * 0.35), max(2, 3)))

        cmap = LinearSegmentedColormap.from_list(
            "trait_cmap", [low_color, mid_color, high_color]
        )

        pivot = cor_df.pivot(
            index="Trait" if False else "dummy", columns="Module", values="cor"
        )
        if pivot.shape[0] == 0:
            pivot = pd.DataFrame({m: [0] for m in mods_plot})

        im = ax.imshow(
            pivot.values, cmap=cmap, aspect="auto", vmin=-plot_max, vmax=plot_max
        )

        for i, mod in enumerate(mods_plot):
            mod_data = cor_df[cor_df["Module"] == mod]
            for j, (_, row) in enumerate(mod_data.iterrows()):
                val = row["cor"]
                _clr = "white" if abs(val) > plot_max * 0.5 else "black"  # noqa: F841
                if label and "sig" in cor_df.columns:
                    txt = row.get("sig", "")
                    if txt:
                        ax.text(
                            i,
                            j,
                            txt,
                            ha="center",
                            va="center",
                            fontsize=text_size,
                            color=text_color,
                            fontweight="bold",
                        )

        ax.set_xticks(range(n_mods))
        ax.set_xticklabels(mods_plot, rotation=45, ha="right", fontsize=9)
        ax.set_yticks([])
        ax.set_title(str(tname), fontsize=11, fontweight="bold")

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Correlation", fontsize=9)

        plot_list[tname] = fig

    if combine and len(plot_list) > 1:
        import matplotlib.gridspec as gridspec

        n_plots = len(plot_list)
        fig2 = plt.figure(figsize=(6, 2 + n_plots * 2))
        outer_gs = gridspec.GridSpec(
            n_plots + 1, 1, height_ratios=[1] * n_plots + [0.15], hspace=0.05
        )

        for i, (tname, fig) in enumerate(plot_list.items()):
            sub_ax = fig2.add_subplot(outer_gs[i])
            old_ax = fig.axes[0]
            im2 = old_ax.images[0]
            sub_ax.imshow(
                im2.get_array(),
                cmap=im2.cmap,
                aspect="auto",
                vmin=im2.get_clim()[0],
                vmax=im2.get_clim()[1],
            )
            sub_ax.set_title(tname, fontsize=10, fontweight="bold")
            sub_ax.set_xticks(old_ax.get_xticks())
            sub_ax.set_xticklabels(
                old_ax.get_xticklabels(), rotation=45, ha="right", fontsize=7
            )
            sub_ax.set_yticks([])

        fig2.suptitle(
            "Module-Trait Correlation", fontsize=12, fontweight="bold", y=0.995
        )
        plt.tight_layout(pad=0.5)

        if save_path:
            fig2.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
            print(f"Saved: {save_path}")
        return fig2
    elif len(plot_list) == 1:
        fig = list(plot_list.values())[0]
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
            print(f"Saved: {save_path}")
        return fig

    return plot_list


# ======================================================================== #
# Enrichment Plotting Functions
# ======================================================================== #


def enrichr_bar_plot(
    enrichr_results: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    top_n: int = 10,
    group_by: str = "database",
    color: str = "#0072B2",
    save_path: str = None,
) -> plt.Figure:
    """
    Bar plot of Enrichr results.

    Replicates R's EnrichrBarPlot function.

    Parameters
    ----------
    enrichr_results : DataFrame or dict
        Output from run_enrichr() or run_enrichr_modules()
    top_n : int
        Top terms to show per group
    group_by : str
        Grouping column ('database' or 'module')
    color : str
        Bar color
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    _setup_publication_style()

    if isinstance(enrichr_results, dict):
        parts = []
        for mod, df in enrichr_results.items():
            if len(df) > 0:
                df_copy = df.copy()
                df_copy["module"] = mod
                parts.append(df_copy)
        if not parts:
            print("No enrichment results.")
            return None
        df = pd.concat(parts, ignore_index=True)
    else:
        df = enrichr_results.copy()

    if len(df) == 0:
        print("No data for bar plot.")
        return None

    groups = df[group_by].unique()

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(8, max(4, len(groups) * 3)), squeeze=False
    )

    for i, grp in enumerate(groups):
        ax = axes[i][0]
        grp_data = (
            df[df[group_by] == grp].nlargest(top_n, columns="combined_score")
            if "combined_score" in df.columns
            else df[df[group_by] == grp].head(top_n)
        )

        y_pos = np.arange(len(grp_data))
        ax.barh(
            y_pos,
            -np.log10(grp_data["pvalue"]),
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(grp_data["term"].str[:50], fontsize=7)
        ax.set_xlabel(r"$-\log_{10}$(P-value)", fontsize=10)
        ax.set_title(str(grp), fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    if save_path:
        try:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Could not save bar plot (insufficient data): {e}")
            plt.close(fig)
            return None

    return fig


def enrichr_dot_plot(
    enrichr_results: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
    top_n: int = 15,
    group_by: str = "database",
    size_col: str = "overlap",
    save_path: str = None,
) -> plt.Figure:
    """
    Dot plot of Enrichr results.

    Replicates R's EnrichrDotPlot function.
    Shows significance (-log10 p-value), gene ratio (x-axis), and overlap count (dot size).

    Parameters
    ----------
    enrichr_results : DataFrame or dict
        Enrichment results
    top_n : int
        Top terms per group
    group_by : str
        Grouping column
    size_col : str
        Column for dot sizing ('overlap' or 'n_genes')
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    _setup_publication_style()

    if isinstance(enrichr_results, dict):
        parts = []
        for mod, df in enrichr_results.items():
            if len(df) > 0:
                df_copy = df.copy()
                df_copy["module"] = mod
                parts.append(df_copy)
        if not parts:
            return None
        df = pd.concat(parts, ignore_index=True)
    else:
        df = enrichr_results.copy()

    if len(df) == 0:
        return None

    groups = sorted(df[group_by].unique())

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(9, max(4, len(groups) * 3.5)), squeeze=False
    )

    for i, grp in enumerate(groups):
        ax = axes[i][0]
        grp_data = df[df[group_by] == grp].copy()
        grp_data = (
            grp_data.nlargest(top_n, columns="combined_score")
            if "combined_score" in df.columns
            else grp_data.head(top_n)
        )

        if len(grp_data) == 0:
            continue

        grp_data[size_col] = pd.to_numeric(grp_data[size_col], errors="coerce").fillna(
            0
        )
        grp_data["n_genes"] = pd.to_numeric(
            grp_data["n_genes"], errors="coerce"
        ).fillna(1)
        grp_data["gene_ratio"] = grp_data[size_col] / grp_data["n_genes"]

        sizes = grp_data[size_col].fillna(1)
        sizes = pd.to_numeric(sizes, errors="coerce").fillna(1)
        sizes = (sizes / sizes.max() * 200 + 20).clip(lower=20, upper=300)

        _scatter = ax.scatter(
            grp_data["gene_ratio"],
            -np.log10(grp_data["pvalue"]),  # noqa: F841
            s=sizes,
            c="#0072B2",
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
        )

        labels = grp_data["term"].str[:45]
        for j, (_, row) in enumerate(grp_data.iterrows()):
            ax.annotate(
                labels.iloc[j],
                (row["gene_ratio"], -np.log10(row["pvalue"])),
                fontsize=6,
                ha="left",
                va="bottom",
                xytext=(3, 3),
                textcoords="offset points",
            )

        ax.set_xlabel("Gene Ratio", fontsize=10)
        ax.set_ylabel(r"$-\log_{10}$(P-value)", fontsize=10)
        ax.set_title(str(grp), fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    if save_path:
        try:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Could not save dot plot (insufficient data): {e}")
            plt.close(fig)
            return None

    return fig


# ======================================================================== #
# Module Preservation Plot
# ======================================================================== #


def plot_module_preservation(
    preserv_df: pd.DataFrame,
    z_thresholds: Dict[str, float] = {"high": 10, "moderate": 5, "weak": 2},
    save_path: str = None,
) -> plt.Figure:
    """
    Bar plot of module preservation statistics.

    Parameters
    ----------
    preserv_df : DataFrame
        Output from module_preservation()
    z_thresholds : dict
        Z-summary thresholds for preservation categories
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    _setup_publication_style()

    if len(preserv_df) == 0:
        print("No preservation data.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(preserv_df) * 0.35)))

    colors_map = {
        "highly preserved": "#009E73",
        "moderately preserved": "#56B4E9",
        "weakly preserved": "#F0E442",
        "non-preserved": "#D55E00",
        "error": "grey",
    }
    bar_colors = [colors_map.get(p, "grey") for p in preserv_df["preservation"]]

    ax = axes[0]
    y_pos = np.arange(len(preserv_df))
    ax.barh(
        y_pos,
        preserv_df["Zsummary"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(
        x=z_thresholds["weak"], color="#D55E00", linestyle="--", alpha=0.7, linewidth=1
    )
    ax.axvline(
        x=z_thresholds["moderate"],
        color="#0072B2",
        linestyle="--",
        alpha=0.7,
        linewidth=1,
    )
    ax.axvline(
        x=z_thresholds["high"], color="#009E73", linestyle="--", alpha=0.7, linewidth=1
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(preserv_df["module"].values, fontsize=9)
    ax.set_xlabel("Z-summary", fontsize=11)
    ax.set_title("Module Preservation (Z-summary)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    valid_mr = preserv_df[preserv_df["medianRank"].notna()]
    y_pos2 = np.arange(len(valid_mr))
    ax2.barh(
        y_pos2,
        valid_mr["medianRank"],
        color=[
            bar_colors[i]
            for i, row in preserv_df.iterrows()
            if row["medianRank"] == row["medianRank"]
        ],
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(valid_mr["module"].values, fontsize=9)
    ax2.set_xlabel("Median Rank", fontsize=11)
    ax2.set_title("Module Preservation (Median Rank)", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# Helper Functions
# ======================================================================== #


def compute_module_umap(
    adata: AnnData,
    n_hubs: int = 10,
    exclude_grey: bool = True,
    genes_use: List[str] = None,
    n_neighbors: int = 25,
    metric: str = "cosine",
    spread: float = 1.0,
    min_dist: float = 0.4,
    supervised: bool = False,
    random_state: int = 42,
    wgcna_name: str = None,
) -> AnnData:
    """
    Compute UMAP embedding for genes based on TOM, using hub genes as features.

    Python equivalent of R's RunModuleUMAP function.
    Runs UMAP on a TOM submatrix where rows are non-grey module genes and
    columns are the top hub genes per module. The resulting UMAP coordinates
    are stored in ``adata.uns['hdWGCNA'][wgcna_name]['module_umap']`` as a
    DataFrame with columns [UMAP1, UMAP2, gene, module, color, hub, kME].

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results (TOM, modules_df)
    n_hubs : int
        Number of top hub genes per module to use as UMAP features
    exclude_grey : bool
        Whether to exclude the grey module (default True, matches R)
    genes_use : list of str
        Optional custom gene list to use for the UMAP, must already be
        present in modules_df. Matches R's genes_use parameter.
    n_neighbors : int
        UMAP n_neighbors parameter (default 25, matches R)
    metric : str
        Distance metric for UMAP (default: 'cosine', matches R)
    spread : float
        UMAP spread parameter (default 1.0, matches R)
    min_dist : float
        UMAP min_dist parameter (default 0.4, matches R)
    supervised : bool
        Whether to use supervised UMAP with module labels as target
        (default False, matches R)
    random_state : int
        Random seed for reproducibility
    wgcna_name : str
        Name of hdWGCNA experiment

    Returns
    -------
    AnnData
        Updated AnnData with module_umap stored in uns
    """
    wn = _get_wgcna_name(adata, wgcna_name)
    wd = adata.uns["hdWGCNA"][wn]
    modules_df = wd.get("modules_df")
    TOM_arr, tom_genes = _get_tom_similarity(wd)

    if TOM_arr is None:
        raise ValueError("No TOM found. Run construct_network() first.")
    if modules_df is None:
        raise ValueError("No module data found. Run construct_network() first.")

    mods = sorted(set(modules_df["module"].unique()) - {"grey"})

    kme_cols_expected = [f"kME_{m}" for m in mods]
    missing_kme = [c for c in kme_cols_expected if c not in modules_df.columns]
    if len(missing_kme) == len(kme_cols_expected):
        raise ValueError(
            "Eigengene-based connectivity (kME) not found. "
            "Did you run module_eigengenes and module_connectivity?"
        )

    if exclude_grey:
        working_modules = modules_df[modules_df["module"] != "grey"].copy()
    else:
        working_modules = modules_df.copy()
        if "grey" not in mods:
            mods = sorted(set(modules_df["module"].unique()))

    hub_genes_list = []
    for cur_mod in mods:
        mod_df = working_modules[working_modules["module"] == cur_mod].copy()
        kme_col = f"kME_{cur_mod}"
        if kme_col not in mod_df.columns:
            kme_cols = [c for c in mod_df.columns if "kME" in c.lower()]
            if kme_cols:
                kme_col = kme_cols[0]
            else:
                continue
        mod_df = mod_df.nlargest(n_hubs, columns=kme_col)
        hub_genes_list.extend(mod_df["gene_name"].values.tolist())

    hub_genes = list(dict.fromkeys(hub_genes_list))

    selected_genes = working_modules["gene_name"].tolist()

    if genes_use is not None:
        selected_genes = [g for g in selected_genes if g in genes_use]

    gene_to_tom_idx = {g: i for i, g in enumerate(tom_genes)}

    sel_idx = [gene_to_tom_idx[g] for g in selected_genes if g in gene_to_tom_idx]
    hub_idx = [gene_to_tom_idx[g] for g in hub_genes if g in gene_to_tom_idx]

    if len(sel_idx) == 0 or len(hub_idx) == 0:
        raise ValueError("No valid genes found for module UMAP.")

    feature_mat = TOM_arr[np.ix_(sel_idx, hub_idx)]

    if umap_lib is None:
        raise ImportError(
            "umap-learn is required for compute_module_umap. "
            "Install with: pip install umap-learn"
        )

    actual_n_neighbors = min(n_neighbors, len(sel_idx) - 1)
    if actual_n_neighbors < 2:
        actual_n_neighbors = 2

    umap_kwargs = dict(
        n_neighbors=actual_n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric=metric,
        spread=spread,
        random_state=random_state,
        n_epochs=200,
        init="spectral",
        transform_seed=42,
        verbose=False,
    )

    if supervised:
        gene_to_module = dict(zip(modules_df["gene_name"], modules_df["module"]))
        y_labels = np.array([gene_to_module.get(g, "grey") for g in selected_genes])
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_labels)
        reducer = umap_lib.UMAP(**umap_kwargs)
        embedding = reducer.fit_transform(feature_mat, y=y_encoded)
    else:
        reducer = umap_lib.UMAP(**umap_kwargs)
        embedding = reducer.fit_transform(feature_mat)

    plot_df = pd.DataFrame(
        {
            "UMAP1": embedding[:, 0],
            "UMAP2": embedding[:, 1],
            "gene": selected_genes,
        }
    )

    gene_to_module = dict(zip(modules_df["gene_name"], modules_df["module"]))
    gene_to_color = dict(zip(modules_df["gene_name"], modules_df["color"]))
    plot_df["module"] = plot_df["gene"].map(gene_to_module).fillna("grey")
    plot_df["color"] = plot_df["gene"].map(gene_to_color).fillna("grey")
    hub_set = set(hub_genes)
    plot_df["hub"] = plot_df["gene"].apply(lambda x: "hub" if x in hub_set else "other")

    kme_dfs = []
    for cur_mod in mods:
        cur = working_modules[working_modules["module"] == cur_mod].copy()
        kme_col = f"kME_{cur_mod}"
        if kme_col not in cur.columns:
            kme_cols_alt = [c for c in cur.columns if "kME" in c.lower()]
            if kme_cols_alt:
                kme_col = kme_cols_alt[0]
            else:
                continue
        cur_kme = cur[["gene_name", kme_col]].copy()
        cur_kme.columns = ["gene_name", "kME"]
        xmin, xmax = cur_kme["kME"].min(), cur_kme["kME"].max()
        if xmax != xmin:
            cur_kme["kME"] = (cur_kme["kME"] - xmin) / (xmax - xmin)
        else:
            cur_kme["kME"] = 0.5
        kme_dfs.append(cur_kme)

    if kme_dfs:
        kme_all = pd.concat(kme_dfs, ignore_index=True)
        kme_all = kme_all.drop_duplicates(subset="gene_name", keep="first")
        gene_kme_map = dict(zip(kme_all["gene_name"], kme_all["kME"]))
    else:
        gene_kme_map = {}

    plot_df["kME"] = plot_df["gene"].map(gene_kme_map).fillna(0.0)

    wd["module_umap"] = plot_df
    adata.uns["hdWGCNA"][wn] = wd

    print(
        f"compute_module_umap complete: {plot_df.shape[0]} genes, "
        f"{(plot_df['hub'] == 'hub').sum()} hub genes, {len(mods)} modules"
    )

    return adata


# ======================================================================== #
# ModuleDotPlot - Seurat-style DotPlot for hMEs by group
# ======================================================================== #


def module_dot_plot(
    adata: AnnData,
    features: str = "hMEs",
    group_by: str = "cell_type",
    exclude_grey: bool = True,
    col_min: float = -2.5,
    col_max: float = 2.5,
    dot_scale: float = 6,
    scale_by: str = "radius",
    rotate_x_labels: bool = True,
    x_label_rotation: int = 45,
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Seurat-style DotPlot for module eigengenes grouped by cell type.

    Replicates R's DotPlot(seurat_obj, features=mods, group.by='celltype')
    with scale_color_gradient2(high='red', mid='grey95', low='blue').

    Dot color = average (z-scored) module eigengene per group.
    Dot size  = fraction of cells with positive eigengene per group.

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    features : str
        'hMEs', 'MEs', 'scores', or 'average'
    group_by : str
        Column in adata.obs for grouping cells
    exclude_grey : bool
        Exclude grey module
    col_min : float
        Minimum scaled average for color mapping
    col_max : float
        Maximum scaled average for color mapping
    dot_scale : float
        Scale factor for dot sizes
    scale_by : str
        'radius' or 'size' for dot scaling
    rotate_x_labels : bool
        Rotate x-axis labels (like Seurat's RotatedAxis)
    x_label_rotation : int
        Rotation angle for x-axis labels
    wgcna_name : str
        Experiment name
    save_path : str
        Save path

    Returns
    -------
    Figure
    """
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)

    me_key = {
        "hMEs": "hMEs",
        "MEs": "MEs",
        "scores": "module_scores",
        "average": "avg_module_expr",
    }.get(features, "hMEs")
    MEs = wd.get(me_key)

    if MEs is None:
        raise ValueError(f"No {features} found. Run module_eigengenes() first.")

    if isinstance(MEs, np.ndarray):
        mod_names = wd.get("module_names", [f"M{i}" for i in range(MEs.shape[1])])
        MEs = pd.DataFrame(
            MEs, columns=mod_names, index=adata.obs_names[: MEs.shape[0]]
        )

    if exclude_grey:
        grey_cols = [c for c in MEs.columns if c.lower() == "grey"]
        if grey_cols:
            MEs = MEs.drop(columns=grey_cols)

    mods = list(MEs.columns)
    if len(mods) == 0:
        raise ValueError("No modules to plot after excluding grey.")

    if group_by not in adata.obs.columns:
        raise ValueError(
            f"Column '{group_by}' not found in adata.obs. "
            f"Available: {list(adata.obs.columns)}"
        )

    _cell_groups = adata.obs[group_by].astype(str).values  # noqa: F841

    common_idx = MEs.index.intersection(adata.obs_names)
    if len(common_idx) == 0:
        MEs = MEs.copy()
        MEs.index = adata.obs_names[: MEs.shape[0]]
        common_idx = MEs.index

    MEs_aligned = MEs.loc[common_idx]
    groups_aligned = adata.obs.loc[common_idx, group_by].astype(str).values

    unique_groups = sorted(set(groups_aligned))
    n_groups = len(unique_groups)
    n_mods = len(mods)

    avg_exp = pd.DataFrame(
        np.zeros((n_groups, n_mods)), index=unique_groups, columns=mods
    )
    pct_exp = pd.DataFrame(
        np.zeros((n_groups, n_mods)), index=unique_groups, columns=mods
    )

    for gi, grp in enumerate(unique_groups):
        mask = groups_aligned == grp
        n_cells_grp = mask.sum()
        if n_cells_grp == 0:
            continue
        grp_mes = MEs_aligned.loc[common_idx[mask]]
        for mod in mods:
            vals = grp_mes[mod].values
            avg_exp.loc[grp, mod] = np.nanmean(vals)
            pct_exp.loc[grp, mod] = np.sum(vals > 0) / len(vals)

    avg_scaled = avg_exp.copy()
    for mod in mods:
        col_vals = avg_exp[mod].values
        std_val = np.nanstd(col_vals, ddof=1)
        if std_val > 0:
            avg_scaled[mod] = (col_vals - np.nanmean(col_vals)) / std_val
        else:
            avg_scaled[mod] = 0.0
    avg_scaled = avg_scaled.clip(lower=col_min, upper=col_max)

    plot_data = []
    for gi, grp in enumerate(unique_groups):
        for mi, mod in enumerate(mods):
            plot_data.append(
                {
                    "group": grp,
                    "module": mod,
                    "avg_scaled": avg_scaled.loc[grp, mod],
                    "pct_exp": pct_exp.loc[grp, mod],
                }
            )
    plot_df = pd.DataFrame(plot_data)

    cmap = LinearSegmentedColormap.from_list(
        "blue_grey95_red", ["#0000FF", "#F2F2F2", "#FF0000"], N=256
    )

    fig_width = max(3.5, n_mods * 0.8 + 1.5)
    fig_height = max(2.5, n_groups * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    norm_color = plt.Normalize(vmin=col_min, vmax=col_max)

    if scale_by == "radius":
        sizes = plot_df["pct_exp"].values * dot_scale**2
    else:
        sizes = plot_df["pct_exp"].values * dot_scale**2 * np.pi

    scatter = ax.scatter(
        plot_df["module"],
        plot_df["group"],
        c=plot_df["avg_scaled"],
        cmap=cmap,
        norm=norm_color,
        s=sizes,
        edgecolors="none",
        linewidths=0.5,
        zorder=3,
    )

    for gi, grp in enumerate(unique_groups):
        for mi, mod in enumerate(mods):
            row = plot_df[(plot_df["group"] == grp) & (plot_df["module"] == mod)]
            if len(row) == 0:
                continue
            pct = row["pct_exp"].values[0]
            if pct > 0:
                ax.scatter(mod, grp, s=0, edgecolors="none", zorder=2)

    ax.set_xlim(-0.5, n_mods - 0.5)
    ax.set_ylim(n_groups - 0.5, -0.5)

    ax.set_xticks(range(n_mods))
    ax.set_xticklabels(
        mods,
        fontsize=10,
        ha="right" if rotate_x_labels else "center",
        rotation_mode="anchor",
    )
    if rotate_x_labels:
        plt.setp(
            ax.get_xticklabels(),
            rotation=x_label_rotation,
            ha="right",
            rotation_mode="anchor",
        )

    ax.set_yticks(range(n_groups))
    ax.set_yticklabels(unique_groups, fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label("Average hME (scaled)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    pct_vals = np.array([0.25, 0.5, 0.75, 1.0])
    if scale_by == "radius":
        legend_sizes = pct_vals * dot_scale**2
    else:
        legend_sizes = pct_vals * dot_scale**2 * np.pi

    legend_elements = []
    for pv, sz in zip(pct_vals, legend_sizes):
        legend_elements.append(
            plt.scatter(
                [],
                [],
                s=sz,
                c="grey",
                edgecolors="black",
                linewidths=0.5,
                label=f"{int(pv * 100)}%",
            )
        )

    _leg = ax.legend(
        handles=legend_elements,
        title="Percent\nexpressed",
        loc="upper left",
        bbox_to_anchor=(1.25, 1.0),
        frameon=True,
        fontsize=8,
        title_fontsize=8,
        labelspacing=1.2,
        borderpad=0.8,
        handletextpad=0.5,
    )

    plt.tight_layout(pad=0.5)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
        print(f"Saved: {save_path}")

    return fig


# ======================================================================== #
# ModuleRadarPlot - Radar/spider plot for module eigengenes by group
# ======================================================================== #


def module_radar_plot(
    adata: AnnData,
    group_by: str = None,
    barcodes: list = None,
    features: str = "hMEs",
    exclude_grey: bool = True,
    fill: bool = True,
    draw_points: bool = False,
    axis_label_size: float = 4,
    grid_label_size: float = 4,
    ncols: int = 4,
    combine: bool = True,
    wgcna_name: str = None,
    save_path: str = None,
) -> Union[plt.Figure, Dict[str, plt.Figure]]:
    """
    Radar/spider plot for module eigengenes grouped by a metadata column.

    Replicates R's ModuleRadarPlot function.
    Each module gets its own radar plot where each axis represents a group
    and the radial distance shows the mean module eigengene (clamped >= 0).

    Parameters
    ----------
    adata : AnnData
        AnnData with hdWGCNA results
    group_by : str
        Column in adata.obs for grouping cells. If None, uses first categorical column.
    barcodes : list
        Subset of cell barcodes to use. If None, uses all cells.
    features : str
        'hMEs', 'MEs', 'scores', or 'average'
    exclude_grey : bool
        Exclude grey module
    fill : bool
        Fill the radar polygon with module color
    draw_points : bool
        Draw points at axis endpoints
    axis_label_size : float
        Font size for axis (group) labels
    grid_label_size : float
        Font size for grid value labels
    ncols : int
        Number of columns in combined layout
    combine : bool
        Combine into single figure (True) or return dict of figures
    wgcna_name : str
        Experiment name
    save_path : str
        Save path

    Returns
    -------
    Figure or dict of Figures
    """
    _setup_publication_style()
    wd = _get_wd(adata, wgcna_name)

    me_key = {
        "hMEs": "hMEs",
        "MEs": "MEs",
        "scores": "module_scores",
        "average": "avg_module_expr",
    }.get(features, "hMEs")
    MEs = wd.get(me_key)

    if MEs is None:
        raise ValueError(f"No {features} found. Run module_eigengenes() first.")

    if isinstance(MEs, np.ndarray):
        mod_names = wd.get("module_names", [f"M{i}" for i in range(MEs.shape[1])])
        MEs = pd.DataFrame(
            MEs, columns=mod_names, index=adata.obs_names[: MEs.shape[0]]
        )

    if exclude_grey:
        grey_cols = [c for c in MEs.columns if c.lower() == "grey"]
        if grey_cols:
            MEs = MEs.drop(columns=grey_cols)

    modules_df = wd.get("modules_df")
    if modules_df is None:
        raise ValueError("No module data found.")

    mods = list(MEs.columns)
    if len(mods) == 0:
        raise ValueError("No modules to plot after excluding grey.")

    mod_colors = {}
    for m in mods:
        row = modules_df[modules_df["module"] == m]
        if len(row) > 0 and "color" in row.columns:
            mod_colors[m] = _to_mpl_color(row["color"].values[0])
        else:
            mod_colors[m] = "#0072B2"

    if group_by is None:
        cat_cols = [
            c for c in adata.obs.columns if adata.obs[c].dtype.name == "category"
        ]
        if cat_cols:
            group_by = cat_cols[0]
        else:
            group_by = adata.obs.columns[0]

    if group_by not in adata.obs.columns:
        raise ValueError(f"Column '{group_by}' not found in adata.obs.")

    common_idx = MEs.index.intersection(adata.obs_names)
    if len(common_idx) == 0:
        MEs = MEs.copy()
        MEs.index = adata.obs_names[: MEs.shape[0]]
        common_idx = MEs.index

    MEs_aligned = MEs.loc[common_idx].copy()
    groups_aligned = adata.obs.loc[common_idx, group_by].astype(str).values

    if barcodes is not None:
        barcode_set = set(barcodes)
        valid_barcodes = [b for b in common_idx if b in barcode_set]
        if len(valid_barcodes) == 0:
            raise ValueError("No valid barcodes found in the data.")
        MEs_aligned = MEs_aligned.loc[valid_barcodes]
        groups_aligned = adata.obs.loc[valid_barcodes, group_by].astype(str).values

    MEs_aligned["_group"] = groups_aligned

    avg_df = MEs_aligned.groupby("_group")[mods].mean()
    avg_df = avg_df.clip(lower=0)

    clusters = list(avg_df.index)

    plot_data = {}
    for mod in mods:
        plot_data[mod] = avg_df[mod].values

    n_mods = len(mods)
    n_clusters = len(clusters)

    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    max_val = max(avg_df.values.max(), 0.01)

    grid_levels = [0.25, 0.5, 0.75, 1.0]
    grid_values = [gl * max_val for gl in grid_levels]

    def _draw_single_radar(mod_name, values, color, ax):
        vals_closed = list(values) + [values[0]]

        for i, (angle, cluster) in enumerate(zip(angles, clusters)):
            ax.plot(
                [0, max_val * 1.05 * np.cos(angle)],
                [0, max_val * 1.05 * np.sin(angle)],
                color="#CCCCCC",
                linewidth=0.5,
                zorder=1,
            )
            label_r = max_val * 1.22
            lx = label_r * np.cos(angle)
            ly = label_r * np.sin(angle)
            deg = np.degrees(angle)
            if 90 < deg < 270:
                rot = deg - 180 + 90
            else:
                rot = deg + 90
            rot = rot % 360
            if rot > 180:
                rot -= 360
            ax.text(
                lx,
                ly,
                cluster,
                ha="center",
                va="center",
                fontsize=axis_label_size,
                rotation=rot,
                zorder=5,
            )

        for gv in grid_values:
            circle_x = [gv * np.cos(a) for a in angles_closed]
            circle_y = [gv * np.sin(a) for a in angles_closed]
            ax.plot(circle_x, circle_y, color="#DDDDDD", linewidth=0.5, zorder=1)

        poly_x = [v * np.cos(a) for v, a in zip(vals_closed, angles_closed)]
        poly_y = [v * np.sin(a) for v, a in zip(vals_closed, angles_closed)]

        if fill:
            ax.fill(poly_x, poly_y, color=color, alpha=0.3, zorder=2)
        ax.plot(poly_x, poly_y, color=color, linewidth=1.5, zorder=3)

        if draw_points:
            for v, a in zip(values, angles):
                ax.scatter(v * np.cos(a), v * np.sin(a), color=color, s=15, zorder=4)

        for i, (gv, gl) in enumerate(zip(grid_values, grid_levels)):
            ax.text(
                0.02,
                gv,
                f"{gl:.0%}",
                fontsize=grid_label_size,
                color="#999999",
                ha="left",
                va="center",
                zorder=5,
            )

        ax.set_xlim(-max_val * 1.4, max_val * 1.4)
        ax.set_ylim(-max_val * 1.4, max_val * 1.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(mod_name, fontsize=11, fontweight="bold", ha="center", pad=8)

    if combine:
        nrows = (n_mods + ncols - 1) // ncols
        fig_width = ncols * 3.0
        fig_height = nrows * 3.2
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_width, fig_height), subplot_kw=dict(polar=False)
        )

        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        for idx, mod in enumerate(mods):
            row, col = idx // ncols, idx % ncols
            ax = axes[row, col]
            _draw_single_radar(mod, plot_data[mod], mod_colors[mod], ax)

        total_cells = nrows * ncols
        for idx in range(n_mods, total_cells):
            row, col = idx // ncols, idx % ncols
            axes[row, col].axis("off")

        plt.tight_layout(pad=0.5)

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
            print(f"Saved: {save_path}")

        return fig
    else:
        fig_list = {}
        for mod in mods:
            fig_single, ax_single = plt.subplots(
                figsize=(3, 3), subplot_kw=dict(polar=False)
            )
            _draw_single_radar(mod, plot_data[mod], mod_colors[mod], ax_single)
            plt.tight_layout(pad=0.3)
            fig_list[mod] = fig_single

        return fig_list


def mod_color_lookup(modules_df: pd.DataFrame, mod_name: str) -> str:
    """Lookup module color."""
    row = modules_df[modules_df["module"] == mod_name]
    if len(row) > 0 and "color" in row.columns:
        return row["color"].values[0]
    return "#0072B2"


# ======================================================================== #
# Convenience: generate all standard plots
# ======================================================================== #


def generate_all_plots(
    adata: AnnData,
    output_dir: str,
    wgcna_name: str = None,
    formats: List[str] = ["pdf"],
) -> Dict[str, str]:
    """
    Generate all standard hdWGCNA visualization plots.

    Parameters
    ----------
    adata : AnnData
        AnnData with complete hdWGCNA results
    output_dir : str
        Directory to save plots
    wgcna_name : str
        Experiment name
    formats : list
        Output formats ('pdf', 'svg', 'png')

    Returns
    -------
    dict
        Mapping of plot type to file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = {}

    _wd = _get_wd(adata, wgcna_name)  # noqa: F841

    try:
        _f = os.path.join(output_dir, "soft_powers")
        for fmt in formats:
            plot_soft_powers(adata, wgcna_name=wgcna_name, save_path=f".{fmt}")
        generated["soft_powers"] = f".{formats[0]}"
    except Exception as e:
        print(f"Skipping soft_powers: {e}")

    try:
        _f = os.path.join(output_dir, "dendrogram")
        for fmt in formats:
            plot_dendrogram(adata, wgcna_name=wgcna_name, save_path=f".{fmt}")
        generated["dendrogram"] = f".{formats[0]}"
    except Exception as e:
        print(f"Skipping dendrogram: {e}")

    try:
        _f = os.path.join(output_dir, "module_feature_plot")
        for fmt in formats:
            module_feature_plot(adata, wgcna_name=wgcna_name, save_path=f".{fmt}")
        generated["module_feature_plot"] = f".{formats[0]}"
    except Exception as e:
        print(f"Skipping module_feature_plot: {e}")

    try:
        _f = os.path.join(output_dir, "kme_bars")  # noqa: F841
        for fmt in formats:
            plot_kmes(adata, wgcna_name=wgcna_name, save_path=f".{fmt}")
        generated["kme_bars"] = f".{formats[0]}"
    except Exception as e:
        print(f"Skipping kme_bars: {e}")

    try:
        _f = os.path.join(output_dir, "correlogram")
        for fmt in formats:
            module_correlogram(adata, wgcna_name=wgcna_name, save_path=f".{fmt}")
        generated["correlogram"] = f".{formats[0]}"
    except Exception as e:
        print(f"Skipping correlogram: {e}")

    print(f"\nGenerated {len(generated)} plots in {output_dir}")
    return generated
