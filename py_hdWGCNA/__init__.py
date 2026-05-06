"""
py-hdWGCNA: Pure-Python re-implementation of R's hdWGCNA package.

A complete WGCNA (Weighted Gene Co-expression Network Analysis) toolkit
for single-cell RNA-seq data, following the hdWGCNA methodology.

Structure:
  - hdWGCNA.py: Main HDWGCNA class with pipeline methods
  - core.py: SetupForWGCNA gene selection
  - metacells.py: ConstructMetacells
  - network.py: TestSoftPowers + ConstructNetwork
  - modules.py: ModuleEigengenes + ModuleConnectivity
  - analysis.py: DME analysis + module-trait correlation (computation only)
  - enrichment.py: Enrichr API integration (computation only)
  - projection.py: ProjectModules + ModulePreservation (computation only)
  - tf_network.py: TF network construction and regulon analysis (computation only)
  - tf_plotting.py: TF visualization functions
  - plotting.py: ALL visualization functions (14 total)
  - utils.py: Helper functions

Usage:
    from py_hdWGCNA import HDWGCNA

    hdw = HDWGCNA()
    hdw.setup_for_wgcna(adata, ...)
    hdw.test_soft_powers(adata, powers=range(2, 15))
    hdw.construct_network(adata, soft_power=6)
    hdw.module_eigengenes(adata)
"""

__version__ = "0.1.0"
__author__ = "py-hdWGCNA Team"

from .hdWGCNA import HDWGCNA
from .core import setup_for_wgcna
from .metacells import metacells_by_groups, normalize_metacells
from .network import test_soft_powers, construct_network
from .modules import module_eigengenes, module_connectivity, reset_module_names
from .analysis import find_dmes, find_all_dmes, module_trait_correlation, overlap_modules_degs, module_expr_score, avg_module_expr
from .enrichment import run_enrichr, run_enrichr_modules
from .projection import project_modules, module_preservation
from .tf_network import (
    generate_motif_data,
    construct_tf_network,
    assign_tf_regulons,
    regulon_scores,
    get_tf_target_genes,
    module_regulatory_network,
    overlap_modules_motifs,
    get_tf_network,
    get_tf_regulons,
    get_regulon_scores,
    find_differential_regulons,
    run_enrichr_regulons,
    get_enrichr_regulon_table,
)
from .tf_plotting import (
    tf_network_plot,
    regulon_bar_plot,
    module_regulatory_network_plot,
    module_regulatory_heatmap,
    plot_differential_regulons,
)
from .utils import (
    load_r_outputs,
    load_test_data,
    benchmark_compare,
    format_benchmark_report,
)
from .plotting import (
    plot_soft_powers,
    module_feature_plot,
    plot_dendrogram,
    plot_kmes,
    module_correlogram,
    module_network_plot,
    hub_gene_network_plot,
    module_umap_plot,
    compute_module_umap,
    plot_dmes_volcano,
    plot_dmes_lollipop,
    plot_module_trait_correlation,
    enrichr_bar_plot,
    enrichr_dot_plot,
    plot_module_preservation,
    module_dot_plot,
    module_radar_plot,
    mod_color_lookup,
    generate_all_plots,
    module_corr_network,
    overlap_dot_plot,
    overlap_bar_plot,
    motif_overlap_bar_plot,
    do_hub_gene_heatmap,
    module_topology_heatmap,
    module_topology_barplot,
    plot_module_preservation_lollipop,
)

__all__ = [
    "HDWGCNA",
    "setup_for_wgcna",
    "metacells_by_groups",
    "normalize_metacells",
    "test_soft_powers",
    "construct_network",
    "module_eigengenes",
    "module_connectivity",
    "reset_module_names",
    "find_dmes",
    "find_all_dmes",
    "module_trait_correlation",
    "overlap_modules_degs",
    "module_expr_score",
    "avg_module_expr",
    "run_enrichr",
    "run_enrichr_modules",
    "project_modules",
    "module_preservation",
    "generate_motif_data",
    "construct_tf_network",
    "assign_tf_regulons",
    "regulon_scores",
    "get_tf_target_genes",
    "module_regulatory_network",
    "overlap_modules_motifs",
    "get_tf_network",
    "get_tf_regulons",
    "get_regulon_scores",
    "find_differential_regulons",
    "run_enrichr_regulons",
    "get_enrichr_regulon_table",
    "tf_network_plot",
    "regulon_bar_plot",
    "module_regulatory_network_plot",
    "module_regulatory_heatmap",
    "plot_differential_regulons",
    "load_r_outputs",
    "load_test_data",
    "benchmark_compare",
    "format_benchmark_report",
    "plot_soft_powers",
    "module_feature_plot",
    "plot_dendrogram",
    "plot_kmes",
    "module_correlogram",
    "module_network_plot",
    "hub_gene_network_plot",
    "module_umap_plot",
    "compute_module_umap",
    "plot_dmes_volcano",
    "plot_dmes_lollipop",
    "plot_module_trait_correlation",
    "enrichr_bar_plot",
    "enrichr_dot_plot",
    "plot_module_preservation",
    "module_dot_plot",
    "module_radar_plot",
    "mod_color_lookup",
    "generate_all_plots",
    "module_corr_network",
    "overlap_dot_plot",
    "overlap_bar_plot",
    "motif_overlap_bar_plot",
    "do_hub_gene_heatmap",
    "module_topology_heatmap",
    "module_topology_barplot",
    "plot_module_preservation_lollipop",
]
