"""
Demo: TF Regulatory Network Analysis — strictly following R tf_network.Rmd

Generates exactly the plots shown in the Rmd tutorial:
  regulon_barplot, TFnetwork_default, TFnetwork_depths, TFnetwork_multi_pos_vs_neg,
  TFnetwork_cor_vs_gain, TFnetwork_customize, module_network_heatmap,
  module_network_heatmap_delta, module_network_plot, module_network_plot_delta,
  module_network_plot_layouts
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from py_hdWGCNA import (
    setup_for_wgcna, metacells_by_groups, normalize_metacells,
    test_soft_powers, construct_network, module_eigengenes, module_connectivity,
    generate_motif_data, construct_tf_network, assign_tf_regulons, regulon_scores,
    module_regulatory_network, overlap_modules_motifs,
    tf_network_plot, regulon_bar_plot,
    module_regulatory_network_plot, module_regulatory_heatmap,
    find_differential_regulons, plot_differential_regulons,
    compute_module_umap,
)
from py_hdWGCNA.core import set_dat_expr
from py_hdWGCNA.utils import check_wgcna_name, get_hdWGCNA_data, set_hdWGCNA_data
from py_hdWGCNA.tf_network import get_tf_regulons


def combine_figs(figs, save_path):
    """Combine multiple matplotlib figures side-by-side into one image using PIL."""
    from PIL import Image
    import io

    images = []
    for fig in figs:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()

    # Scale all images to the same height
    max_h = max(im.height for im in images)
    resized = []
    for im in images:
        if im.height != max_h:
            new_w = int(im.width * max_h / im.height)
            im = im.resize((new_w, max_h), Image.LANCZOS)
        resized.append(im)

    total_w = sum(im.width for im in resized)
    combined = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in resized:
        combined.paste(im, (x, 0))
        x += im.width

    combined.save(save_path, dpi=(300, 300))
    print(f"Saved: {os.path.basename(save_path)}")
    for f in figs:
        plt.close(f)


# ------------------------------------------------------------------ #
# Output directory — clean first
# ------------------------------------------------------------------ #
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tf_output")
os.makedirs(OUT_DIR, exist_ok=True)
for f in os.listdir(OUT_DIR):
    if f.endswith(".png"):
        os.remove(os.path.join(OUT_DIR, f))

# ------------------------------------------------------------------ #
# 1. Load data
# ------------------------------------------------------------------ #
print("=" * 60)
print("STEP 1: Load data")
print("=" * 60)

h5ad_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results", "py_hdWGCNA_notebook", "data", "test_seurat.h5ad"
)
adata = sc.read_h5ad(h5ad_path)
print(f"  {adata.n_obs} cells x {adata.n_vars} genes")

# ------------------------------------------------------------------ #
# 2. Standard WGCNA pipeline
# ------------------------------------------------------------------ #
print("\nSTEP 2: WGCNA pipeline")
wn = "HDWGCNA"
adata = setup_for_wgcna(adata, gene_select="fraction", fraction=0.05, wgcna_name=wn)
adata = metacells_by_groups(adata, group_by=["cell_type", "Sample"], k=25,
                            max_shared=10, min_cells=50, reduction="X_pca", wgcna_name=wn)
adata = normalize_metacells(adata, wgcna_name=wn)
adata = set_dat_expr(adata, group_name="OPC", group_by="cell_type", use_metacells=True, wgcna_name=wn)
adata = test_soft_powers(adata, network_type="signed", cor_method="pearson", wgcna_name=wn)
adata = construct_network(adata, minModuleSize=25, deepSplit=2, mergeCutHeight=0.25, wgcna_name=wn)
adata = module_eigengenes(adata, group_by_vars="Sample", harmonize=True, wgcna_name=wn)
adata = module_connectivity(adata, wgcna_name=wn)

# ------------------------------------------------------------------ #
# 3. Motif data → update genes_use → re-run SetDatExpr
# ------------------------------------------------------------------ #
print("\nSTEP 3: Motif data + gene list update")
adata = generate_motif_data(adata, source="all", seed=42, wgcna_name=wn)

wd = adata.uns["hdWGCNA"][wn]
modules_df = wd["modules_df"]
tf_genes = list(wd["motif_info"]["gene_name"].unique())
nongrey_genes = modules_df[modules_df["module"] != "grey"]["gene_name"].tolist()
genes_use = list(dict.fromkeys(tf_genes + nongrey_genes))
wd["genes_use"] = genes_use
adata = set_hdWGCNA_data(adata, wd, wn)
adata = set_dat_expr(adata, group_name="OPC", group_by="cell_type", use_metacells=True, wgcna_name=wn)

# ------------------------------------------------------------------ #
# 4. Add simulated TFs (demo only)
# ------------------------------------------------------------------ #
print("\nSTEP 4: Add simulated TFs for visualization")
np.random.seed(42)
wd = adata.uns["hdWGCNA"][wn]
existing_tf_genes = set(wd["motif_info"]["gene_name"].tolist())
existing_genes = set(modules_df["gene_name"].tolist())
non_grey = modules_df[modules_df["module"] != "grey"].copy()
kME_cols = [c for c in modules_df.columns if c.startswith("kME_")]

sim_tfs = []
for mod in non_grey["module"].unique():
    mod_genes = non_grey[non_grey["module"] == mod]
    if kME_cols:
        mod_genes = mod_genes.copy()
        mod_genes["max_kME"] = mod_genes[kME_cols].max(axis=1)
        hubs = mod_genes.nlargest(8, "max_kME")["gene_name"].tolist()
    else:
        hubs = mod_genes["gene_name"].tolist()[:8]
    for g in hubs:
        if g not in existing_tf_genes:
            sim_tfs.append(g)
sim_tfs = sim_tfs[:25]

new_rows, new_targets = [], {}
for i, tf in enumerate(sim_tfs):
    motif_id = f"SIM{str(i).zfill(6)}"
    n_targets = np.random.randint(15, 40)
    targets = np.random.choice(
        [g for g in existing_genes if g != tf],
        size=min(n_targets, len(existing_genes) - 1), replace=False,
    ).tolist()
    new_rows.append({"motif_id": motif_id, "motif_name": tf, "gene_name": tf, "n_targets": len(targets)})
    new_targets[tf] = targets
    col = np.zeros(len(modules_df), dtype=int)
    for t in targets:
        col[list(modules_df["gene_name"]).index(t)] = 1
    wd["motif_matrix"][motif_id] = col

wd["motif_info"] = pd.concat([wd["motif_info"], pd.DataFrame(new_rows)], ignore_index=True)
wd["motif_targets"].update(new_targets)
adata = set_hdWGCNA_data(adata, wd, wn)
print(f"  {len(sim_tfs)} simulated TFs added, total: {len(wd['motif_info'])}")

# ------------------------------------------------------------------ #
# 5. TF network + regulons (R parameters)
# ------------------------------------------------------------------ #
print("\nSTEP 5: Construct TF Network + Regulons")
model_params = {"objective": "reg:squarederror", "max_depth": 1, "eta": 0.1,
                "nthread": 4, "alpha": 0.5, "verbosity": 0}
adata = construct_tf_network(adata, model_params=model_params, nfold=5, wgcna_name=wn)
adata = assign_tf_regulons(adata, strategy="A", reg_thresh=0.01, n_tfs=10, wgcna_name=wn)

# Add simulated TF entries to regulons for visualization
wd = adata.uns["hdWGCNA"][wn]
sim_rows = []
for tf in sim_tfs:
    for gene in wd["motif_targets"].get(tf, []):
        if gene in existing_genes and gene != tf:
            sim_rows.append({"tf": tf, "gene": gene, "Gain": np.random.uniform(0.5, 5.0),
                             "Cover": np.random.uniform(10, 100), "Frequency": np.random.randint(1, 20),
                             "Cor": np.random.uniform(-0.3, 0.5)})
if sim_rows:
    df = pd.DataFrame(sim_rows)
    df["reg_score"] = df["Gain"] * np.sign(df["Cor"])
    wd["tf_regulons"] = pd.concat([wd["tf_regulons"], df], ignore_index=True)
    adata = set_hdWGCNA_data(adata, wd, wn)

# Regulon scores
adata = regulon_scores(adata, target_type="positive", cor_thresh=0.05, wgcna_name=wn)
adata = regulon_scores(adata, target_type="negative", cor_thresh=-0.05, wgcna_name=wn)

# ------------------------------------------------------------------ #
# Identify TFs for plots
# ------------------------------------------------------------------ #
tf_regulons = get_tf_regulons(adata, wgcna_name=wn)
top_tfs = tf_regulons.groupby("tf")["Gain"].sum().nlargest(10).index.tolist()
cur_tf = top_tfs[0]
cur_tfs = top_tfs[:3]
all_tfs_set = set(tf_regulons["tf"].unique())

# Hub genes in same module as cur_tf (for customize plot)
cur_mod = non_grey[non_grey["gene_name"] == cur_tf]["module"].iloc[0] if cur_tf in non_grey["gene_name"].values else non_grey["module"].iloc[0]
cur_mod_genes = non_grey[non_grey["module"] == cur_mod]["gene_name"].tolist()

print(f"\n  TFs: {len(all_tfs_set)}, selected: {cur_tf}, multi: {cur_tfs}")

# ================================================================== #
# PLOTS — matching R tf_network.Rmd exactly
# ================================================================== #
print("\n" + "=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

# --- 1. RegulonBarPlot: p1 | p2 (Rmd line 266-269) ---
print("\n  [1/11] regulon_barplot.png")
p1 = regulon_bar_plot(adata, selected_tf=cur_tf, figsize=(6, 5), save_path=None)
p2 = regulon_bar_plot(adata, selected_tf=top_tfs[1], cutoff=0.15, figsize=(6, 5), save_path=None)
combine_figs([p1, p2], os.path.join(OUT_DIR, "regulon_barplot.png"))

# --- 2. TFNetworkPlot default (Rmd line 351) ---
print("  [2/11] TFnetwork_default.png")
p1 = tf_network_plot(adata, selected_tfs=[cur_tf], figsize=(8, 8), save_path=None)
p1.savefig(os.path.join(OUT_DIR, "TFnetwork_default.png"), dpi=300, bbox_inches="tight")
print(f"Saved: TFnetwork_default.png")
plt.close(p1)

# --- 3. TFNetworkPlot depth 1|2|3 (Rmd line 401-403) ---
print("  [3/11] TFnetwork_depths.png")
figs = []
for d in [1, 2, 3]:
    fig = tf_network_plot(adata, selected_tfs=[cur_tf], depth=d, no_labels=True,
                          figsize=(6, 6), save_path=None)
    figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "TFnetwork_depths.png"))

# --- 4. TFNetworkPlot multi-TF target_type (Rmd line 420-437) ---
print("  [4/11] TFnetwork_multi_pos_vs_neg.png")
figs = []
for tt in ["positive", "both", "negative"]:
    fig = tf_network_plot(adata, selected_tfs=cur_tfs, target_type=tt,
                          label_tfs_depth=0, depth=1, figsize=(6, 6), save_path=None)
    title = {"positive": "positive targets", "both": "pos & neg targets", "negative": "negative targets"}[tt]
    fig.axes[0].set_title(title, fontsize=12, fontweight="bold")
    figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "TFnetwork_multi_pos_vs_neg.png"))

# --- 5. TFNetworkPlot edge_weight Cor vs Gain (Rmd line 447-457) ---
print("  [5/11] TFnetwork_cor_vs_gain.png")
figs = []
for ew, title in [("Cor", "edge_weight='Cor'"), ("Gain", "edge_weight='Gain'")]:
    fig = tf_network_plot(adata, selected_tfs=[cur_tf], edge_weight=ew, cutoff=0.05,
                          figsize=(6, 6), save_path=None)
    fig.axes[0].set_title(title, fontsize=12, fontweight="bold")
    figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "TFnetwork_cor_vs_gain.png"))

# --- 6. TFNetworkPlot customize: TFs_only | custom labels | custom colors (Rmd line 475-493) ---
print("  [6/11] TFnetwork_customize.png")
figs = []
# p1: TFs only
fig = tf_network_plot(adata, selected_tfs=[cur_tf], tfs_only=True, figsize=(6, 6), save_path=None)
fig.axes[0].set_title("TFs only", fontsize=12, fontweight="bold")
figs.append(fig)
# p2: custom gene labels
fig = tf_network_plot(adata, selected_tfs=[cur_tf], label_tfs_depth=0, label_genes=cur_mod_genes,
                      figsize=(6, 6), save_path=None)
fig.axes[0].set_title("Custom gene labels", fontsize=12, fontweight="bold")
figs.append(fig)
# p3: custom colors
fig = tf_network_plot(adata, selected_tfs=[cur_tf], label_tfs_depth=0,
                      high_color="hotpink", mid_color="grey98", low_color="seagreen",
                      node_colors=["grey30", "grey60", "grey90"], figsize=(6, 6), save_path=None)
fig.axes[0].set_title("Custom colors", fontsize=12, fontweight="bold")
figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "TFnetwork_customize.png"))

# --- 7. ModuleRegulatoryHeatmap: positive | negative (Rmd line 816-823) ---
print("  [7/11] module_network_heatmap.png")
figs = []
for feat, hc in [("positive", "orange2"), ("negative", "dodgerblue")]:
    fig = module_regulatory_heatmap(adata, feature=feat, high_color=hc, figsize=(6, 5), save_path=None)
    figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "module_network_heatmap.png"))

# --- 8. ModuleRegulatoryHeatmap: delta TFs_only | delta all_genes (Rmd line 849-855) ---
print("  [8/11] module_network_heatmap_delta.png")
figs = []
fig = module_regulatory_heatmap(adata, feature="delta", dendrogram=False, figsize=(6, 5), save_path=None)
fig.axes[-1].set_title("TFs only", fontsize=11, fontweight="bold")
figs.append(fig)
fig = module_regulatory_heatmap(adata, feature="delta", tfs_only=False, max_val=5,
                                dendrogram=False, figsize=(6, 5), save_path=None)
fig.axes[-1].set_title("All target genes", fontsize=11, fontweight="bold")
figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "module_network_heatmap_delta.png"))

# --- 9. ModuleRegulatoryNetworkPlot: positive | negative (Rmd line 880-885) ---
print("  [9/11] module_network_plot.png")
figs = []
for feat, hc in [("positive", "orange2"), ("negative", "dodgerblue")]:
    fig = module_regulatory_network_plot(adata, feature=feat, high_color=hc, figsize=(6, 6), save_path=None)
    figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "module_network_plot.png"))

# --- 10. ModuleRegulatoryNetworkPlot: delta (Rmd line 932-943) ---
print("  [10/11] module_network_plot_delta.png")
figs = []
fig = module_regulatory_network_plot(adata, feature="delta", cutoff=0.5, max_val=1.5,
                                     figsize=(6, 6), save_path=None)
figs.append(fig)
fig = module_regulatory_network_plot(adata, feature="delta", cutoff=0.5, max_val=1.5,
                                     layout="stress", loops=False,
                                     label_modules=False, figsize=(6, 6), save_path=None)
figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "module_network_plot_delta.png"))

# --- 11. ModuleRegulatoryNetworkPlot: circle | stress layouts (Rmd line 895-901) ---
print("  [11/11] module_network_plot_layouts.png")
figs = []
fig = module_regulatory_network_plot(adata, layout="circle", loops=False, figsize=(6, 6), save_path=None)
fig.axes[0].set_title("layout='circle'", fontsize=12, fontweight="bold")
figs.append(fig)
fig = module_regulatory_network_plot(adata, layout="stress", loops=False, figsize=(6, 6), save_path=None)
fig.axes[0].set_title("layout='stress'", fontsize=12, fontweight="bold")
figs.append(fig)
combine_figs(figs, os.path.join(OUT_DIR, "module_network_plot_layouts.png"))

# ------------------------------------------------------------------ #
# 12. Differential Regulon Analysis (Rmd line 663-719)
# ------------------------------------------------------------------ #
print("\n  [12/13] differential_regulons.png")
np.random.seed(42)
cells = adata.obs_names.tolist()
np.random.shuffle(cells)
group1 = cells[:len(cells) // 2]
group2 = cells[len(cells) // 2:]
dregs = find_differential_regulons(adata, barcodes1=group1, barcodes2=group2, wgcna_name=wn)
print(f"  Differential regulons: {len(dregs)} TFs analyzed")
fig = plot_differential_regulons(adata, dregs, wgcna_name=wn, save_path=None)
fig.savefig(os.path.join(OUT_DIR, "differential_regulons.png"), dpi=300, bbox_inches="tight")
print(f"Saved: differential_regulons.png")
plt.close(fig)

# ------------------------------------------------------------------ #
# 13. ModuleRegulatoryNetworkPlot with UMAP background (Rmd line 930-943)
# ------------------------------------------------------------------ #
print("  [13/13] module_network_plot_umap.png")
adata = compute_module_umap(adata, n_hubs=5, n_neighbors=15, min_dist=0.1, wgcna_name=wn)
fig = module_regulatory_network_plot(
    adata, feature="delta", cutoff=0.5, max_val=1.5,
    umap_background=True, label_modules=False, figsize=(8, 8), save_path=None,
)
fig.savefig(os.path.join(OUT_DIR, "module_network_plot_umap.png"), dpi=300, bbox_inches="tight")
print(f"Saved: module_network_plot_umap.png")
plt.close(fig)

# ------------------------------------------------------------------ #
# Summary
# ------------------------------------------------------------------ #
print("\n" + "=" * 60)
print("COMPLETE")
print("=" * 60)
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".png"):
        size_kb = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f} ({size_kb:.0f} KB)")
