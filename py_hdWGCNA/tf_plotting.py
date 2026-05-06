"""
Transcription Factor (TF) visualization functions for py-hdWGCNA.

Pure-matplotlib re-implementation of R hdWGCNA TF plotting functions:
  - tf_network_plot: Network of TFs and target genes
  - regulon_bar_plot: Bar plot of top target genes for a TF
  - module_regulatory_network_plot: Module-level regulatory network
  - module_regulatory_heatmap: Regulatory network heatmap
"""

from __future__ import annotations

import os
from typing import List, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch
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


def _to_mpl_color(color_name: str) -> str:
    """Convert R color name to matplotlib hex color."""
    _R_COLORS = {
        "orange2": "#FF7F00",
        "dodgerblue": "#1E90FF",
        "mediumpurple2": "#9F79EE",
        "darkorchid4": "#68228B",
        "turquoise": "#40E0D0",
        "midnightblue": "#191970",
        "grey60": "#999999",
        "greenyellow": "#ADFF2F",
        "lightyellow": "#FFFFE0",
        "lightcyan": "#E0FFFF",
        "grey90": "#E6E6E6",
    }
    if color_name in _R_COLORS:
        return _R_COLORS[color_name]
    # Handle R greyN / grayN colors (e.g. grey98 = 98% grey)
    import re
    m = re.match(r"^(?:grey|gray)(\d+)$", color_name, re.IGNORECASE)
    if m:
        pct = int(m.group(1)) / 100.0
        val = int(round(pct * 255))
        return f"#{val:02x}{val:02x}{val:02x}"
    try:
        return mcolors.to_hex(color_name)
    except ValueError:
        return color_name


def _setup_publication_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _hierarchical_layout(all_nodes, edge_list, selected_tfs, all_tfs_set):
    """
    Hierarchical layout mimicking R sparse_stress:
    selected TFs in center, other TFs in ring, genes clustered around parent TF.
    """
    selected_in = [n for n in all_nodes if n in selected_tfs]
    other_tfs = [n for n in all_nodes if n in all_tfs_set and n not in selected_tfs]
    genes = [n for n in all_nodes if n not in all_tfs_set]

    # Build parent map: for each node, which TF connects to it (first edge wins)
    parent_map = {}
    for src, tgt in edge_list:
        if tgt not in parent_map and src in all_tfs_set:
            parent_map[tgt] = src
    for tf in selected_tfs:
        parent_map[tf] = None  # root nodes

    # Group children by parent TF
    children_map = {}
    for node in all_nodes:
        parent = parent_map.get(node)
        if parent is not None:
            children_map.setdefault(parent, []).append(node)

    pos = {}
    n_sel = len(selected_in)

    # Phase 1: Place selected TFs in center cluster
    if n_sel == 1:
        pos[selected_in[0]] = np.array([0.0, 0.0])
    else:
        R_sel = 0.22
        for i, tf in enumerate(selected_in):
            angle = 2 * np.pi * i / n_sel - np.pi / 2
            pos[tf] = np.array([R_sel * np.cos(angle), R_sel * np.sin(angle)])

    # Phase 2: Place other TFs in outer ring
    n_other = len(other_tfs)
    if n_other > 0:
        R_other = 0.55
        for i, tf in enumerate(other_tfs):
            angle = 2 * np.pi * i / n_other - np.pi / 4
            pos[tf] = np.array([R_other * np.cos(angle), R_other * np.sin(angle)])

    # Phase 3: Place genes clustered around their parent TF
    # Sort parents by number of children (biggest clusters first, more room)
    sorted_parents = sorted(children_map.keys(),
                            key=lambda p: -len(children_map[p]))

    for parent in sorted_parents:
        kids = children_map[parent]
        parent_pos = pos.get(parent, np.array([0.0, 0.0]))
        n_kids = len(kids)
        if n_kids == 0:
            continue

        # Radius grows with number of children — more generous spacing
        R_gene = 0.15 + 0.006 * n_kids

        # Find angle pointing away from graph center
        dist_from_center = np.linalg.norm(parent_pos)
        if dist_from_center < 1e-9:
            base_angle = 0.0
        else:
            base_angle = np.arctan2(parent_pos[1], parent_pos[0])

        # Spread children in an arc around the parent
        for j, gene in enumerate(kids):
            if n_kids == 1:
                angle = base_angle
            else:
                # Wider arc for more children
                spread = min(np.pi * 1.5, 0.15 * n_kids)
                angle = base_angle + spread * (j / (n_kids - 1) - 0.5)
            # Small jitter to avoid overlap
            jitter_r = 0.012 * (hash(gene) % 100 / 100 - 0.5)
            jitter_a = 0.06 * (hash(gene) % 73 / 73 - 0.5)
            pos[gene] = parent_pos + np.array([
                (R_gene + jitter_r) * np.cos(angle + jitter_a),
                (R_gene + jitter_r) * np.sin(angle + jitter_a)
            ])

    # Handle orphan nodes
    for node in all_nodes:
        if node not in pos:
            pos[node] = np.array([0.0, 0.0])

    return pos


def _igraph_layout(node_names, edge_list, edge_weights=None, layout="graphopt"):
    """Compute layout using igraph (produces ggraph-like results)."""
    import igraph as ig

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    idx_edges = [(name_to_idx[u], name_to_idx[v]) for u, v in edge_list]

    g = ig.Graph(n=len(node_names), edges=idx_edges, directed=True)
    g.vs["name"] = list(node_names)

    if layout == "graphopt":
        lay = g.layout("graphopt", niter=500)
    elif layout == "drl":
        lay = g.layout("drl")
    elif layout == "stress":
        w = None
        if edge_weights:
            w = [max(abs(edge_weights[i]), 1e-6) for i in range(len(edge_list))]
        lay = g.layout("kk", weights=w)
    elif layout == "circle":
        lay = g.layout("circle")
    elif layout == "kk":
        w = None
        if edge_weights:
            w = [max(abs(edge_weights[i]), 1e-6) for i in range(len(edge_list))]
        lay = g.layout("kk", weights=w)
    else:
        lay = g.layout("fr", niter=300)

    coords = np.array(lay.coords)
    if coords.max(axis=0).any():
        coords = 2 * (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-9) - 1
    return {node_names[i]: coords[i] for i in range(len(node_names))}


def _draw_network_edges(ax, pos, edge_list, edge_weights, cmap, norm,
                         width_scale=2.5, min_width=0.3, base_alpha=0.7, rad=0.15):
    """Draw each curved edge individually with per-edge alpha and width."""
    abs_max = max(abs(norm.vmin), abs(norm.vmax), 1e-9)

    # Group edges by pair to assign different curvature offsets
    pair_counts = {}
    for i, (u, v) in enumerate(edge_list):
        pair_key = (min(u, v), max(u, v))
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    pair_idx = {}
    for i, (u, v) in enumerate(edge_list):
        pair_key = (min(u, v), max(u, v))
        idx = pair_idx.get(pair_key, 0)
        pair_idx[pair_key] = idx + 1

        w = edge_weights[i]
        color = cmap(norm(w))
        alpha = base_alpha * (abs(w) / abs_max)
        alpha = max(0.12, min(alpha, 0.85))
        width = max(abs(w) * width_scale, min_width)

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        if u == v:
            # Self-loop
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=3,head_width=2",
                    color=color, lw=width, alpha=alpha,
                    connectionstyle=f"arc3,rad=1.2",
                ),
            )
            continue

        # Multiple edges between same pair get different curvature
        n_pairs = pair_counts[pair_key]
        if n_pairs > 1:
            curv = rad * (1 + 0.5 * (idx - (n_pairs - 1) / 2))
        else:
            curv = rad

        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={curv}",
            arrowstyle="-|>,head_length=4,head_width=2.5",
            color=color, lw=width, alpha=alpha, zorder=1,
            shrinkA=6, shrinkB=6,
        )
        ax.add_patch(arrow)


# ------------------------------------------------------------------ #
# TF Network Plot
# ------------------------------------------------------------------ #

def tf_network_plot(
    adata: AnnData,
    selected_tfs: List[str],
    depth: int = 2,
    edge_weight: str = "Cor",
    cutoff: float = 0.01,
    color_cutoff: float = 0.75,
    target_type: str = "both",
    use_regulons: bool = True,
    label_genes: List[str] = None,
    label_tfs_depth: int = 1,
    no_labels: bool = False,
    tfs_only: bool = False,
    high_color: str = "orange2",
    mid_color: str = "white",
    low_color: str = "dodgerblue",
    node_colors: List[str] = None,
    figsize: Tuple[float, float] = (10, 10),
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot network of TFs and predicted target genes.

    Re-implements R hdWGCNA::TFNetworkPlot using matplotlib + igraph layouts.
    """
    from adjustText import adjust_text
    from .tf_network import get_tf_target_genes, get_tf_network, get_tf_regulons

    _setup_publication_style()

    if node_colors is None:
        node_colors = ["black", "darkorchid4", "mediumpurple2"]

    # Get network data
    tf_net = get_tf_network(adata, wgcna_name)
    if use_regulons:
        tf_regulons = get_tf_regulons(adata, wgcna_name)
    else:
        tf_regulons = tf_net

    if tf_regulons is None:
        raise ValueError("TF data not found.")

    all_tfs_set = set(tf_regulons["tf"].unique())
    tf_degree_map = tf_regulons.groupby("tf")["gene"].nunique().to_dict()

    # Get target genes with depth
    cur_network = get_tf_target_genes(
        adata, selected_tfs=selected_tfs, depth=depth,
        target_type=target_type, use_regulons=use_regulons,
        wgcna_name=wgcna_name,
    )

    # Get depth of each gene
    gene_depths = cur_network.groupby("gene")["depth"].min().to_dict()
    for tf in selected_tfs:
        gene_depths[tf] = 0

    # Edge weight
    if edge_weight == "Gain":
        cur_network["edge_weight"] = cur_network["Gain"] * np.sign(cur_network["Cor"])
    else:
        cur_network["edge_weight"] = cur_network["Cor"]

    cur_network["edge_weight"] = cur_network["edge_weight"].clip(-color_cutoff, color_cutoff)
    cur_network = cur_network[cur_network["edge_weight"].abs() >= cutoff]

    if tfs_only:
        cur_network = cur_network[cur_network["gene"].isin(all_tfs_set)]

    if len(cur_network) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No nodes in network", ha="center", va="center", fontsize=14)
        return fig

    # Build node list and edge list
    all_nodes = list(dict.fromkeys(
        list(cur_network["tf"].values) + list(cur_network["gene"].values)
    ))
    edge_list = list(zip(cur_network["tf"].values, cur_network["gene"].values))
    edge_weights = cur_network["edge_weight"].values.tolist()

    # Layout: use igraph force-directed layout (R uses sparse_stress via ggraph)
    pos = _igraph_layout(all_nodes, edge_list, edge_weights, layout="kk")

    # Node sizes (R: TF size=5, Gene size=2 in ggplot2 units ≈ small points)
    def node_size(n):
        if n in selected_tfs:
            return 70 + 10 * tf_degree_map.get(n, 5)
        elif n in all_tfs_set:
            return 50 + 8 * tf_degree_map.get(n, 3)
        else:
            return 18

    # Edge color map
    cmap = LinearSegmentedColormap.from_list(
        "custom", [_to_mpl_color(low_color), _to_mpl_color(mid_color), _to_mpl_color(high_color)]
    )
    abs_max = max(abs(w) for w in edge_weights) if edge_weights else 1.0
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    # Draw edges
    _draw_network_edges(ax, pos, edge_list, edge_weights, cmap, norm,
                         width_scale=1.5, min_width=0.15, base_alpha=0.6, rad=0.1)

    # Node color palette
    cp = [_to_mpl_color(c) for c in node_colors]

    # R shapes: selected TF=23 (diamond), other TF=25 (v-triangle), gene=16 (circle)
    # Classify nodes
    selected_in = [n for n in all_nodes if n in selected_tfs]
    other_tf_in = [n for n in all_nodes if n in all_tfs_set and n not in selected_tfs]
    gene_in = [n for n in all_nodes if n not in all_tfs_set]

    # Gene nodes (circles)
    if gene_in:
        xs = [pos[n][0] for n in gene_in]
        ys = [pos[n][1] for n in gene_in]
        colors = [cp[min(gene_depths.get(n, depth), len(cp) - 1)] for n in gene_in]
        sizes = [node_size(n) for n in gene_in]
        ax.scatter(xs, ys, s=sizes, c=colors, marker="o",
                    edgecolors="black", linewidths=0.5, zorder=3)

    # Other TF nodes (downward triangles)
    if other_tf_in:
        xs = [pos[n][0] for n in other_tf_in]
        ys = [pos[n][1] for n in other_tf_in]
        colors = [cp[min(gene_depths.get(n, depth), len(cp) - 1)] for n in other_tf_in]
        sizes = [node_size(n) for n in other_tf_in]
        ax.scatter(xs, ys, s=sizes, c=colors, marker="v",
                    edgecolors="black", linewidths=0.7, zorder=3)

    # Selected TF nodes (diamonds)
    if selected_in:
        xs = [pos[n][0] for n in selected_in]
        ys = [pos[n][1] for n in selected_in]
        colors = [cp[min(gene_depths.get(n, depth), len(cp) - 1)] for n in selected_in]
        sizes = [node_size(n) for n in selected_in]
        ax.scatter(xs, ys, s=sizes, c=colors, marker="D",
                    edgecolors="black", linewidths=1.0, zorder=3)

    # Labels with adjustText
    if not no_labels:
        texts = []
        for node in all_nodes:
            d = gene_depths.get(node, depth)
            # Label all TFs and explicitly requested gene labels
            is_tf = node in all_tfs_set
            is_label_gene = label_genes and node in label_genes
            show = (is_tf and (node in selected_tfs or d <= label_tfs_depth)) or is_label_gene
            if show:
                x, y = pos[node]
                fs = 8 if node in selected_tfs else 6.5
                fw = "bold" if node in selected_tfs else "normal"
                t = ax.text(x, y, node, fontsize=fs, fontstyle="italic",
                            fontweight=fw, ha="center", va="center", zorder=5)
                texts.append(t)
        if texts:
            adjust_text(
                texts, ax=ax, max_move=None, iter_limit=100,
                expand=(1.3, 1.5),
                force_text=(0.6, 0.8),
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.3, alpha=0.4),
            )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02, aspect=20)
    cbar.set_label(edge_weight, fontsize=10)
    cbar.outline.set_linewidth(0.7)

    # Legend (R: diamond=selected, triangle=TF, circle=Gene)
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], marker="D", color="w", markerfacecolor=cp[0],
                       markeredgecolor="black", markersize=8, label="Selected TF"),
        mlines.Line2D([0], [0], marker="v", color="w", markerfacecolor=cp[1],
                       markeredgecolor="black", markersize=8, label="TF"),
        mlines.Line2D([0], [0], marker="o", color="w", markerfacecolor=cp[2],
                       markeredgecolor="black", markersize=6, label="Gene"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", frameon=True,
              fancybox=False, edgecolor="black", fontsize=8, handlelength=1.5)

    ax.set_title("TF Network", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


# ------------------------------------------------------------------ #
# Regulon Bar Plot
# ------------------------------------------------------------------ #

def regulon_bar_plot(
    adata: AnnData,
    selected_tf: str,
    cutoff: float = 0.2,
    top_n: int = None,
    tfs_only: bool = False,
    high_color: str = "orange2",
    mid_color: str = "white",
    low_color: str = "dodgerblue",
    figsize: Tuple[float, float] = (8, 6),
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Bar plot of top target genes for a specific TF.

    Re-implements R hdWGCNA::RegulonBarPlot.
    """
    from .tf_network import get_tf_regulons

    _setup_publication_style()

    tf_regulons = get_tf_regulons(adata, wgcna_name)
    if tf_regulons is None:
        raise ValueError("Regulons not found.")

    if tfs_only:
        all_tfs = set(tf_regulons["tf"].unique())
        tf_regulons = tf_regulons[tf_regulons["gene"].isin(all_tfs)]

    if selected_tf not in tf_regulons["tf"].values:
        raise ValueError(f"TF '{selected_tf}' not found in regulons.")

    # Compute regulatory score
    plot_df = tf_regulons[tf_regulons["tf"] == selected_tf].copy()
    plot_df["score"] = plot_df["Gain"] * np.sign(plot_df["Cor"])
    plot_df = plot_df[plot_df["score"].abs() > cutoff]

    top = plot_df[plot_df["Cor"] > 0].nlargest(top_n or len(plot_df), "Gain")
    bottom = plot_df[plot_df["Cor"] < 0].nsmallest(top_n or len(plot_df), "Gain")
    plot_df = pd.concat([top, bottom]).sort_values("score").reset_index(drop=True)

    if len(plot_df) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No targets for {selected_tf} above cutoff",
                ha="center", va="center")
        return fig

    # Gradient colors (R: scale_fill_gradient2)
    cmap = LinearSegmentedColormap.from_list(
        "bar", [_to_mpl_color(low_color), _to_mpl_color(mid_color), _to_mpl_color(high_color)]
    )
    score_vals = plot_df["score"].values
    abs_max = max(abs(score_vals))
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)
    bar_colors = [cmap(norm(s)) for s in score_vals]

    # Plot
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    y_pos = np.arange(len(plot_df))
    bars = ax.barh(y_pos, score_vals, color=bar_colors, edgecolor="black",
                    linewidth=0.3, height=0.75, zorder=2)

    # Gene labels inside bars (R: geom_text hjust='inward', italic)
    for i, (score, gene) in enumerate(zip(score_vals, plot_df["gene"].values)):
        if score > 0:
            ax.text(score - abs_max * 0.02, i, gene, va="center", ha="right",
                    fontsize=7.5, fontstyle="italic", color="black", zorder=3)
        else:
            ax.text(score + abs_max * 0.02, i, gene, va="center", ha="left",
                    fontsize=7.5, fontstyle="italic", color="black", zorder=3)

    # R-style clean theme
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.tick_params(left=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axvline(x=0, color="black", linewidth=0.8, zorder=1)
    ax.set_xlabel("Regulatory score", fontsize=11)
    ax.set_title(f"{selected_tf} predicted targets", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.2, zorder=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


# ------------------------------------------------------------------ #
# Module Regulatory Network Plot
# ------------------------------------------------------------------ #

def module_regulatory_network_plot(
    adata: AnnData,
    feature: str = "delta",
    tfs_only: bool = True,
    max_val: float = 1.0,
    cutoff: float = 0.0,
    loops: bool = True,
    label_modules: bool = True,
    layout: str = "graphopt",
    umap_background: bool = False,
    focus_source: str = None,
    focus_target: str = None,
    high_color: str = "orange2",
    mid_color: str = "white",
    low_color: str = "dodgerblue",
    figsize: Tuple[float, float] = (10, 10),
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize regulatory network between modules as a graph.

    Re-implements R hdWGCNA::ModuleRegulatoryNetworkPlot.
    """
    from adjustText import adjust_text
    from .tf_network import module_regulatory_network

    _setup_publication_style()

    reg_net = module_regulatory_network(adata, tfs_only=tfs_only, wgcna_name=wgcna_name)

    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")

    # Module color map (R: mod_colors)
    if modules_df is not None:
        non_grey = modules_df[modules_df["module"] != "grey"]
        mod_colors = dict(zip(non_grey["module"], non_grey["color"]))
        mod_sizes = non_grey.groupby("module")["gene_name"].count().to_dict()
    else:
        mod_colors = {}
        mod_sizes = {}

    # Score column
    if feature == "positive":
        reg_net["score"] = reg_net["score_pos"]
    elif feature == "negative":
        reg_net["score"] = reg_net["score_neg"]
    else:
        reg_net["score"] = reg_net["score_pos"] - reg_net["score_neg"]

    reg_net["score"] = reg_net["score"].clip(-max_val, max_val)
    reg_net = reg_net[reg_net["score"].abs() >= cutoff]

    # Focus filtering (R: focus_source, focus_target)
    if focus_source is not None:
        reg_net = reg_net[reg_net["source"].astype(str) == focus_source]
    if focus_target is not None:
        reg_net = reg_net[reg_net["target"].astype(str) == focus_target]

    # Build edge data
    edge_list = []
    edge_weights = []
    for _, row in reg_net.iterrows():
        src, tgt = str(row["source"]), str(row["target"])
        if src == tgt and not loops:
            continue
        edge_list.append((src, tgt))
        edge_weights.append(row["score"])

    if len(edge_list) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No edges above cutoff", ha="center", va="center")
        return fig

    node_names = list(dict.fromkeys(
        [e[0] for e in edge_list] + [e[1] for e in edge_list]
    ))

    # Layout (R: layout='graphopt'|'circle'|'stress'|'drl'|'kk', or 'umap')
    umap_df = None
    if umap_background or layout == "umap":
        umap_df = wd.get("module_umap")
        if umap_df is not None:
            # Compute module centroid positions from UMAP
            umap_pos = {}
            for mod in node_names:
                mod_pts = umap_df[umap_df["module"] == mod]
                if len(mod_pts) > 0:
                    umap_pos[mod] = np.array([mod_pts["UMAP1"].mean(), mod_pts["UMAP2"].mean()])
            # Fill missing with igraph layout
            if len(umap_pos) < len(node_names):
                fallback = _igraph_layout(node_names, edge_list, edge_weights, layout="graphopt")
                for n in node_names:
                    if n not in umap_pos:
                        umap_pos[n] = fallback[n]
            pos = umap_pos
        else:
            pos = _igraph_layout(node_names, edge_list, edge_weights, layout="graphopt")
    else:
        pos = _igraph_layout(node_names, edge_list, edge_weights, layout=layout)

    # Edge color map
    cmap = LinearSegmentedColormap.from_list(
        "reg", [_to_mpl_color(low_color), _to_mpl_color(mid_color), _to_mpl_color(high_color)]
    )
    abs_max = max(abs(w) for w in edge_weights) if edge_weights else 1.0
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    # UMAP background: draw gene points colored by module
    if umap_background and umap_df is not None:
        for _, row in umap_df.iterrows():
            ax.scatter(row["UMAP1"], row["UMAP2"],
                       s=4 + 8 * row.get("kME", 0.5),
                       c=row["color"], alpha=0.3, marker="o", zorder=0, linewidths=0)

    # Draw edges
    _draw_network_edges(ax, pos, edge_list, edge_weights, cmap, norm,
                         width_scale=4.0, min_width=0.5, base_alpha=0.75, rad=0.15)

    # Nodes (R: shape=21 = circle with fill + black border)
    max_size = max(mod_sizes.get(n, 1) for n in node_names) if mod_sizes else 1
    for node in node_names:
        x, y = pos[node]
        size = 300 + 1500 * (mod_sizes.get(node, 1) / max_size)
        color = _to_mpl_color(mod_colors.get(node, "#999999"))
        ax.scatter(x, y, s=size, c=color, marker="o",
                    edgecolors="black", linewidths=1.3, zorder=3)

    # Labels with adjustText
    if label_modules:
        texts = []
        for node in node_names:
            x, y = pos[node]
            t = ax.text(x, y, node, fontsize=10, fontweight="bold",
                        ha="center", va="center", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="black", linewidth=0.6, alpha=0.85))
            texts.append(t)
        adjust_text(texts, ax=ax, max_move=None, iter_limit=80,
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.4, alpha=0.5))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_label = {"positive": "Positive\nRegulatory\nScore",
                  "negative": "Negative\nRegulatory\nScore",
                  "delta": "Pos - Neg\nRegulatory\nScore"}[feature]
    cbar = fig.colorbar(sm, ax=ax, shrink=0.45, pad=0.02, aspect=20)
    cbar.set_label(cbar_label, fontsize=10, rotation=0, labelpad=30)
    cbar.outline.set_linewidth(0.7)

    ax.set_title("Module Regulatory Network", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


# ------------------------------------------------------------------ #
# Module Regulatory Heatmap
# ------------------------------------------------------------------ #

def module_regulatory_heatmap(
    adata: AnnData,
    feature: str = "delta",
    tfs_only: bool = True,
    dendrogram: bool = True,
    max_val: float = 1.0,
    min_val_label: int = 3,
    high_color: str = "orange2",
    mid_color: str = "white",
    low_color: str = "dodgerblue",
    figsize: Tuple[float, float] = (8, 7),
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Visualize module regulatory network as a heatmap.

    Re-implements R hdWGCNA::ModuleRegulatoryHeatmap.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram as scipy_dendrogram
    from scipy.spatial.distance import pdist
    from .tf_network import module_regulatory_network

    _setup_publication_style()

    reg_net = module_regulatory_network(adata, tfs_only=tfs_only, wgcna_name=wgcna_name)

    if feature == "positive":
        reg_net["score"] = reg_net["score_pos"]
        reg_net["n"] = reg_net["n_pos"]
    elif feature == "negative":
        reg_net["score"] = reg_net["score_neg"]
        reg_net["n"] = reg_net["n_neg"]
    else:
        reg_net["score"] = reg_net["score_pos"] - reg_net["score_neg"]
        reg_net["n"] = 0

    reg_net["score"] = reg_net["score"].clip(-max_val, max_val)
    reg_net["label"] = np.where(reg_net["n"] >= min_val_label, reg_net["n"].astype(str), "")

    mods = sorted(reg_net["source"].unique(),
                   key=lambda x: int(x.replace("M", "")) if x.startswith("M") else 0)
    score_matrix = reg_net.pivot(index="target", columns="source", values="score")
    label_matrix = reg_net.pivot(index="target", columns="source", values="label")
    score_matrix = score_matrix.reindex(index=mods, columns=mods).fillna(0)
    label_matrix = label_matrix.reindex(index=mods, columns=mods).fillna("")

    # Dendrogram
    if dendrogram and len(mods) > 2:
        Z = linkage(pdist(score_matrix.values), method="ward")
        order = leaves_list(Z)
        ordered_mods = [mods[i] for i in order]
        score_matrix = score_matrix.reindex(index=ordered_mods, columns=ordered_mods)
        label_matrix = label_matrix.reindex(index=ordered_mods, columns=ordered_mods)

        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 1, height_ratios=[0.2, 1], hspace=0.05)

        ax_dendro = fig.add_subplot(gs[0])
        scipy_dendrogram(Z, labels=mods, ax=ax_dendro, orientation="top",
                         leaf_rotation=90, leaf_font_size=8, color_threshold=0)
        ax_dendro.set_xticklabels([])
        ax_dendro.set_yticks([])
        for spine in ax_dendro.spines.values():
            spine.set_visible(False)

        ax_heat = fig.add_subplot(gs[1])
    else:
        fig, ax_heat = plt.subplots(figsize=figsize)

    # Heatmap (R: geom_tile + scale_fill_gradient2)
    if feature in ("positive", "negative"):
        cmap = LinearSegmentedColormap.from_list(
            "reg_hm", [_to_mpl_color(mid_color), _to_mpl_color(high_color)]
        )
        vmin = 0
    else:
        cmap = LinearSegmentedColormap.from_list(
            "reg_hm",
            [_to_mpl_color(low_color), _to_mpl_color(mid_color), _to_mpl_color(high_color)]
        )
        vmin = -max_val
    vmax = max_val

    im = ax_heat.imshow(
        score_matrix.values, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="auto", interpolation="nearest",
    )

    n_mods = len(score_matrix.columns)
    for i in range(n_mods):
        for j in range(n_mods):
            lbl = label_matrix.iloc[i, j]
            if lbl:
                ax_heat.text(j, i, str(lbl), ha="center", va="center", fontsize=7)

    ax_heat.set_xticks(range(n_mods))
    ax_heat.set_xticklabels(score_matrix.columns, rotation=90, fontsize=9)
    ax_heat.set_yticks(range(n_mods))
    ax_heat.set_yticklabels(score_matrix.index, fontsize=9)
    ax_heat.set_xlabel("Source Module", fontsize=11)
    ax_heat.set_ylabel("Target Module", fontsize=11)

    cbar_label = {"positive": "Positive\nRegulatory\nScore",
                  "negative": "Negative\nRegulatory\nScore",
                  "delta": "Pos - Neg\nRegulatory\nScore"}[feature]
    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.8)
    cbar.set_label(cbar_label, fontsize=10, rotation=0, labelpad=30)
    cbar.outline.set_linewidth(1.0)

    ax_heat.set_title("Module Regulatory Heatmap", fontsize=13, fontweight="bold")
    ax_heat.set_frame_on(True)
    for spine in ax_heat.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


# ------------------------------------------------------------------ #
# Differential Regulon Plot
# ------------------------------------------------------------------ #

def plot_differential_regulons(
    adata: AnnData,
    dregs: pd.DataFrame,
    n_label: int = 10,
    logfc_thresh: float = 0.1,
    lm: bool = True,
    figsize: Tuple[float, float] = (8, 8),
    wgcna_name: str = None,
    save_path: str = None,
) -> plt.Figure:
    """
    Scatter plot of differential regulon results.

    Re-implements R hdWGCNA::PlotDifferentialRegulons.
    """
    from adjustText import adjust_text

    _setup_publication_style()

    wd = _get_wd(adata, wgcna_name)
    modules_df = wd.get("modules_df")

    # Module color map
    if modules_df is not None:
        non_grey = modules_df[modules_df["module"] != "grey"]
        mod_colors = dict(zip(non_grey["module"], non_grey["color"]))
    else:
        mod_colors = {}

    df = dregs.copy()

    # Classify significance
    sig_pos = (df["avg_log2FC_positive"] < 0) & (df["avg_log2FC_negative"] > 0) & \
              ((df["p_val_adj_positive"] <= 0.05) | (df["p_val_adj_negative"] <= 0.05))
    sig_neg = (df["avg_log2FC_positive"] > 0) & (df["avg_log2FC_negative"] < 0) & \
              ((df["p_val_adj_positive"] <= 0.05) | (df["p_val_adj_negative"] <= 0.05))
    df["significant"] = sig_pos | sig_neg

    # DEG classification
    df["is_deg"] = (df["avg_log2FC_deg"].abs() >= logfc_thresh) & (df["p_val_adj_deg"] < 0.05)

    # 4 shape categories
    categories = [
        (~df["significant"] & ~df["is_deg"], "o", 0.5, 0.5, "Non-significant"),
        (~df["significant"] & df["is_deg"], "D", 0.5, 0.5, "Non-sig. DEG"),
        (df["significant"] & ~df["is_deg"], "o", 1.0, 1.0, "Significant"),
        (df["significant"] & df["is_deg"], "D", 1.0, 1.0, "Sig. DEG"),
    ]

    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    for mask, marker, alpha, edge_lw, label in categories:
        subset = df[mask]
        if len(subset) == 0:
            continue
        colors = [mod_colors.get(m, "#999999") for m in subset["module"].fillna("grey")]
        ax.scatter(
            subset["avg_log2FC_positive"],
            subset["avg_log2FC_negative"],
            c=colors, marker=marker, s=40 + 120 * subset["kME"].fillna(0),
            alpha=alpha, edgecolors="black" if edge_lw > 0.5 else "none",
            linewidths=0.8 if edge_lw > 0.5 else 0,
            label=label, zorder=2,
        )

    # Symmetric axis limits
    x = df["avg_log2FC_positive"].values
    y = df["avg_log2FC_negative"].values
    lim = max(abs(x).max(), abs(y).max()) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Quadrant counts
    q_labels = {
        "top-left": ((x < 0) & (y > 0) & df["significant"].values).sum(),
        "top-right": ((x > 0) & (y > 0) & df["significant"].values).sum(),
        "bottom-left": ((x < 0) & (y < 0) & df["significant"].values).sum(),
        "bottom-right": ((x > 0) & (y < 0) & df["significant"].values).sum(),
    }
    offsets = {
        "top-left": (-lim * 0.85, lim * 0.85),
        "top-right": (lim * 0.85, lim * 0.85),
        "bottom-left": (-lim * 0.85, -lim * 0.85),
        "bottom-right": (lim * 0.85, -lim * 0.85),
    }
    for quad, count in q_labels.items():
        ox, oy = offsets[quad]
        ax.text(ox, oy, str(count), fontsize=14, fontweight="bold",
                ha="center", va="center", color="#666666", zorder=1)

    # Reference lines
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.7, zorder=0)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.7, zorder=0)

    # Regression line
    if lm and len(df) > 2:
        mask_valid = np.isfinite(x) & np.isfinite(y)
        if mask_valid.sum() > 2:
            slope, intercept = np.polyfit(x[mask_valid], y[mask_valid], 1)
            x_line = np.linspace(-lim, lim, 100)
            ax.plot(x_line, slope * x_line + intercept, color="grey",
                    linestyle="-", linewidth=1, alpha=0.5, zorder=0)

    # Label top TFs
    sig_df = df[df["significant"]].copy()
    texts = []
    if len(sig_df) > 0:
        up = sig_df.nlargest(n_label, "avg_log2FC_positive")
        down = sig_df.nsmallest(n_label, "avg_log2FC_positive")
        label_df = pd.concat([up, down]).drop_duplicates()
        for _, row in label_df.iterrows():
            t = ax.text(
                row["avg_log2FC_positive"], row["avg_log2FC_negative"],
                row["tf"], fontsize=7, fontstyle="italic", ha="center", va="center",
                zorder=5,
            )
            texts.append(t)

    if texts:
        adjust_text(texts, ax=ax, max_move=None, iter_limit=80,
                    expand=(1.2, 1.4), force_text=(0.5, 0.7),
                    arrowprops=dict(arrowstyle="-", color="grey", lw=0.3, alpha=0.4))

    ax.set_xlabel("Avg. log2(FC), Positive Regulons", fontsize=11)
    ax.set_ylabel("Avg. log2(FC), Negative Regulons", fontsize=11)
    ax.set_title("Differential Regulon Analysis", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig
