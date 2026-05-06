"""
Main HDWGCNA class for single-cell co-expression network analysis.

Pure-Python re-implementation of hdWGCNA (R package), exposed as
a single `HDWGCNA` class for convenient use with AnnData objects.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from anndata import AnnData


class HDWGCNA:
    """
    hdWGCNA-style co-expression network analysis for single-cell data.

    Wraps a pure-Python implementation of hdWGCNA as a stateful analyzer
    operating on an AnnData object. All results are stored in the AnnData
    (``.uns['hdWGCNA']``) so the usual scanpy workflow continues.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix (cells x genes). Expression matrix in
        ``adata.X`` should be normalized counts.

    Attributes
    ----------
    adata : AnnData
        The annotated data matrix with analysis results stored in-place.

    Examples
    --------
    Basic co-expression network analysis:

    >>> hdw = HDWGCNA(adata)
    >>> hdw.setup_for_wgcna(gene_select='fraction', fraction=0.05)
    >>> hdw.metacells_by_groups(group.by=['cell_type', 'Sample'], k=25)
    >>> hdw.normalize_metacells()
    >>> hdw.set_dat_expr(group_name='INH', group_by='cell_type')
    >>> hdw.test_soft_powers(network_type='signed')
    >>> hdw.construct_network()
    >>> hdw.module_eigengenes(group.by.vars='Sample')
    >>> hdw.module_connectivity(group_by='cell_type', group_name='INH')

    Module trait correlation:

    >>> corr_df = hdw.module_trait_correlation(
    ...     traits=['age', 'diagnosis'],
    ...     trait_adata=adata.obs
    ... )
    """

    def __init__(self, adata: AnnData):
        """Initialize the HDWGCNA analyser.

        Parameters
        ----------
        adata : AnnData
            Annotated data matrix (cells x genes). Should contain normalized
            expression data in ``adata.X``, with variable features selected,
            and dimensionality reduction computed.
        """
        from . import core, metacells, network, modules, plotting

        self._core = core
        self._metacells = metacells
        self._network = network
        self._modules = modules
        self._plotting = plotting

        self.adata = adata

    def __repr__(self):
        status = []
        status.append(f"HDWGCNA({self.adata.n_obs} cells x {self.adata.n_vars} genes)")

        if "hdWGCNA" in self.adata.uns:
            active = self.adata.uns["hdWGCNA"].get("active_wgcna", "N/A")
            status.append(f"  active experiment: {active}")

            if active != "N/A" and active in self.adata.uns["hdWGCNA"]:
                wd = self.adata.uns["hdWGCNA"][active]

                if wd.get("setup_complete"):
                    n_genes = wd.get("n_genes", 0)
                    status.append(f"  setup complete: {n_genes} genes")

                if "metacell_obj" in wd:
                    mc_n = wd["metacell_obj"].n_obs
                    status.append(f"  metacells: {mc_n}")

                if "modules_df" in wd:
                    mods = wd["modules_df"]
                    n_mods = len(set(mods["module"])) - (
                        1 if "grey" in set(mods["module"]) else 0
                    )
                    status.append(f"  modules: {n_mods}")

                if "hMEs" in wd:
                    status.append("  hMEs: computed")

                if wd.get("kME_computed"):
                    status.append("  kME: computed")

        return "\n".join(status)

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #

    def setup_for_wgcna(
        self,
        gene_select: str = "fraction",
        fraction: float = 0.05,
        n_genes: int = None,
        genes_use: list = None,
        wgcna_name: str = None,
    ):
        """Set up AnnData object for hdWGCNA analysis.

        Selects genes and initializes hdWGCNA experiment storage.

        Parameters
        ----------
        gene_select : str
            Gene selection method: 'fraction', 'variable', or 'custom'
        fraction : float
            Fraction of cells for gene expression threshold
        n_genes : int
            Number of top variable genes to select
        genes_use : list
            Custom list of genes for 'custom' method
        wgcna_name : str
            Name for this hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._core.setup_for_wgcna(
            self.adata,
            gene_select=gene_select,
            fraction=fraction,
            n_genes=n_genes,
            genes_use=genes_use,
            wgcna_name=wgcna_name,
        )
        return self

    def set_dat_expr(
        self,
        group_name: str | list = None,
        group_by: str = None,
        assay: str = "RNA",
        layer: str = "data",
        use_metacells: bool = True,
        wgcna_name: str = None,
    ):
        """Set up expression matrix for network construction.

        Parameters
        ----------
        group_name : str or list
            Name(s) of cell groups to include
        group_by : str
            Column name containing group labels
        assay : str
            Assay name
        layer : str
            Layer name for expression data
        use_metacells : bool
            Use metacell expression matrix
        wgcna_name : str
            Name of hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._core.set_dat_expr(
            self.adata,
            group_name=group_name,
            group_by=group_by,
            assay=assay,
            layer=layer,
            use_metacells=use_metacells,
            wgcna_name=wgcna_name,
        )
        return self

    # ------------------------------------------------------------------ #
    # Metacells
    # ------------------------------------------------------------------ #

    def metacells_by_groups(
        self,
        group_by: list | str = None,
        reduction: str = "pca",
        k: int = 25,
        max_shared: int = 10,
        ident_group: str = None,
        min_cells: int = 50,
        target_metacells: int = None,
        wgcna_name: str = None,
    ):
        """Construct metacell expression matrices for each cell group.

        Parameters
        ----------
        group_by : list or str
            Column names to group by
        reduction : str
            Dimensionality reduction for KNN
        k : int
            Number of neighbors per metacell
        max_shared : int
            Maximum shared cells between metacells
        ident_group : str
            Column for metacell identity
        min_cells : int
            Minimum cells per group
        target_metacells : int
            Target metacells per group
        wgcna_name : str
            Name of hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._metacells.metacells_by_groups(
            self.adata,
            group_by=group_by,
            reduction=reduction,
            k=k,
            max_shared=max_shared,
            ident_group=ident_group,
            min_cells=min_cells,
            target_metacells=target_metacells,
            wgcna_name=wgcna_name,
        )
        return self

    def normalize_metacells(self, wgcna_name: str = None):
        """Normalize metacell expression matrix (log normalization).

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._core.normalize_metacells(self.adata, wgcna_name)
        return self

    def aggregate_gene_expression(
        self, group_by: str | list = None, method: str = "mean", wgcna_name: str = None
    ):
        """Aggregate gene expression across groups (pseudobulk).

        Parameters
        ----------
        group_by : str or list
            Column(s) to aggregate by
        method : str
            Aggregation: 'mean' or 'sum'
        wgcna_name : str
            Name of hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._metacells.aggregate_gene_expression(
            self.adata, group_by=group_by, method=method, wgcna_name=wgcna_name
        )
        return self

    # ------------------------------------------------------------------ #
    # Network Construction
    # ------------------------------------------------------------------ #

    def test_soft_powers(
        self,
        power_range: list = None,
        network_type: str = "signed",
        cor_method: str = "bicor",
        wgcna_name: str = None,
    ):
        """Test different soft-thresholding powers for scale-free topology fit.

        Parameters
        ----------
        power_range : list
            Powers to test (default: 1-20)
        network_type : str
            'signed', 'unsigned', or 'signed hybrid'
        cor_method : str
            Correlation method: 'bicor', 'pearson', or 'spearman'
        wgcna_name : str
            Name of hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._network.test_soft_powers(
            self.adata,
            power_range=power_range,
            network_type=network_type,
            cor_method=cor_method,
            wgcna_name=wgcna_name,
        )
        return self

    def construct_network(
        self,
        power: int = None,
        tom_name: str = "hdWGCNA_TOM",
        network_type: str = "signed",
        tom_type: str = "signed",
        tom_denom: str = "min",
        minModuleSize: int = 20,
        deepSplit: int = 4,
        pamRespectsDendro: bool = True,
        pamStage: bool = False,
        mergeCutHeight: float = 0.2,
        wgcna_name: str = None,
        **kwargs,
    ):
        """Construct co-expression network using WGCNA approach.

        Parameters
        ----------
        power : int
            Soft-thresholding power (auto-selected if not provided)
        tom_name : str
            TOM file name
        network_type : str
            Network type ('signed', 'unsigned', 'signed hybrid')
        tom_type : str
            TOM type ('signed', 'unsigned')
        tom_denom : str
            TOM denominator ('min', 'max')
        minModuleSize : int
            Minimum module size
        deepSplit : int
            DeepSplit parameter (0-4)
        pamRespectsDendro : bool
            Whether PAM respects dendrogram
        pamStage : bool
            Whether to perform PAM stage
        mergeCutHeight : float
            Cut height for merging
        wgcna_name : str
            Name of hdWGCNA experiment
        **kwargs
            Additional WGCNA parameters

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._network.construct_network(
            self.adata,
            power=power,
            tom_name=tom_name,
            network_type=network_type,
            tom_type=tom_type,
            tom_denom=tom_denom,
            minModuleSize=minModuleSize,
            deepSplit=deepSplit,
            pamRespectsDendro=pamRespectsDendro,
            pamStage=pamStage,
            mergeCutHeight=mergeCutHeight,
            wgcna_name=wgcna_name,
            **kwargs,
        )
        return self

    # ------------------------------------------------------------------ #
    # Module Analysis
    # ------------------------------------------------------------------ #

    def module_eigengenes(
        self,
        group_by_vars: str | list = "Sample",
        harmonize: bool = True,
        n_pcs: int = 30,
        wgcna_name: str = None,
    ):
        """Compute module eigengenes (MEs) in single cells.

        Parameters
        ----------
        group_by_vars : str or list
            Variable(s) to harmonize by
        harmonize : bool
            Apply Harmony correction
        n_pcs : int
            Number of principal components for eigengene computation
        wgcna_name : str
            Name of hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._modules.module_eigengenes(
            self.adata,
            group_by_vars=group_by_vars,
            harmonize=harmonize,
            n_pcs=n_pcs,
            wgcna_name=wgcna_name,
        )
        return self

    def module_connectivity(
        self,
        group_by: str = None,
        group_name: str | list = None,
        use_metacells: bool = False,
        cor_method: str = "bicor",
        sparse: bool = True,
        wgcna_name: str = None,
    ):
        """Compute eigengene-based connectivity (kME).

        Parameters
        ----------
        group_by : str
            Column to subset by
        group_name : str or list
            Group name(s)
        use_metacells : bool
            Use metacell expression data (default False, matching R)
        cor_method : str
            Correlation method: 'pearson' or 'bicor' (default 'bicor', matching R)
        sparse : bool
            Use sparse correlation (forces pearson, matching R corSparse)
        wgcna_name : str
            Name of hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._modules.module_connectivity(
            self.adata,
            group_by=group_by,
            group_name=group_name,
            use_metacells=use_metacells,
            cor_method=cor_method,
            sparse=sparse,
            wgcna_name=wgcna_name,
        )
        return self

    def reassign_modules(
        self,
        harmonized: bool = True,
        auto_reassign: bool = False,
        wgcna_name: str = None,
    ):
        """Reassign features based on kME values.

        Parameters
        ----------
        harmonized : bool
            Use harmonized MEs
        auto_reassign : bool
            Auto-reassign negative kME genes
        wgcna_name : str
            Name of hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._modules.reassign_modules(
            self.adata,
            harmonized=harmonized,
            auto_reassign=auto_reassign,
            wgcna_name=wgcna_name,
        )
        return self

    def reset_module_names(self, new_name: str = "M", wgcna_name: str = None):
        """Reset module names with custom prefix.

        Parameters
        ----------
        new_name : str
            Base name prefix
        wgcna_name : str
            Name of hdWGCNA experiment

        Returns
        -------
        HDWGCNA
            self for chaining.
        """
        self.adata = self._modules.reset_module_names(
            self.adata, new_name=new_name, wgcna_name=wgcna_name
        )
        return self

    # ------------------------------------------------------------------ #
    # Data Accessors
    # ------------------------------------------------------------------ #

    def get_modules(self, wgcna_name: str = None) -> pd.DataFrame:
        """Get module assignment table."""
        return self._modules.get_modules(self.adata, wgcna_name)

    def get_mes(self, harmonized: bool = True, wgcna_name: str = None) -> pd.DataFrame:
        """Get module eigengenes."""
        return self._modules.get_mes(self.adata, harmonized, wgcna_name)

    def get_hub_genes(self, n_hubs: int = 10, wgcna_name: str = None) -> pd.DataFrame:
        """Get top hub genes ranked by kME."""
        return self._modules.get_hub_genes(self.adata, n_hubs, wgcna_name)

    def get_power_table(self, wgcna_name: str = None) -> pd.DataFrame:
        """Get soft power test results table."""
        return self._network.get_power_table(self.adata, wgcna_name)

    def get_tom(self, wgcna_name: str = None) -> np.ndarray:
        """Get Topological Overlap Matrix."""
        return self._network.get_tom(self.adata, wgcna_name)

    def get_metacell_object(self, wgcna_name: str = None) -> AnnData:
        """Get metacell AnnData object."""
        return self._metacells.get_metacell_object(self.adata, wgcna_name)

    def get_wgcna_genes(self, wgcna_name: str = None) -> list:
        """Get the list of genes used for WGCNA analysis."""
        if wgcna_name is None and "hdWGCNA" in self.adata.uns:
            wgcna_name = self.adata.uns["hdWGCNA"].get("active_wgcna", None)
        return self._modules.get_wgcna_genes(self.adata, wgcna_name)

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #

    def plot_soft_powers(
        self,
        selected_power: int = None,
        point_size: float = 50,
        text_size: int = 8,
        plot_connectivity: bool = True,
        wgcna_name: str = None,
        save_path: str = None,
    ):
        """Plot soft power threshold selection results."""
        return self._plotting.plot_soft_powers(
            self.adata,
            selected_power=selected_power,
            point_size=point_size,
            text_size=text_size,
            plot_connectivity=plot_connectivity,
            wgcna_name=wgcna_name,
            save_path=save_path,
        )

    def module_feature_plot(
        self,
        module_names=None,
        reduction="umap",
        features="hMEs",
        point_size=3,
        alpha=1.0,
        restrict_range=True,
        wgcna_name=None,
        save_path=None,
        ncols=3,
    ):
        """Plot module eigengenes on UMAP/tSNE."""
        return self._plotting.module_feature_plot(
            self.adata,
            module_names=module_names,
            reduction=reduction,
            features=features,
            point_size=point_size,
            alpha=alpha,
            restrict_range=restrict_range,
            wgcna_name=wgcna_name,
            save_path=save_path,
            ncols=ncols,
        )

    def plot_dendrogram(
        self,
        group_labels="Module Colors",
        hang=0.03,
        add_guide=True,
        guide_hang=0.05,
        main="",
        wgcna_name=None,
        save_path=None,
    ):
        """Plot gene dendrogram with module colors."""
        return self._plotting.plot_dendrogram(
            self.adata,
            group_labels=group_labels,
            hang=hang,
            add_guide=add_guide,
            guide_hang=guide_hang,
            main=main,
            wgcna_name=wgcna_name,
            save_path=save_path,
        )

    def plot_kmes(
        self, n_hubs=10, text_size=6, ncols=4, wgcna_name=None, save_path=None
    ):
        """Plot kME barplots per module."""
        return self._plotting.plot_kmes(
            self.adata,
            n_hubs=n_hubs,
            text_size=text_size,
            ncols=ncols,
            wgcna_name=wgcna_name,
            save_path=save_path,
        )

    def module_correlogram(
        self,
        features="hMEs",
        exclude_grey=True,
        method="ellipse",
        cmap=None,
        wgcna_name=None,
        save_path=None,
    ):
        """Plot module eigengene correlation heatmap."""
        return self._plotting.module_correlogram(
            self.adata,
            features=features,
            exclude_grey=exclude_grey,
            method=method,
            cmap=cmap,
            wgcna_name=wgcna_name,
            save_path=save_path,
        )

    def module_network_plot(
        self,
        n_inner=10,
        n_outer=15,
        n_conns=500,
        mods="all",
        outdir="ModuleNetworks",
        wgcna_name=None,
        plot_size=(6, 6),
        edge_alpha=0.25,
        edge_width=1,
        vertex_label_cex=1,
        vertex_size=6,
    ):
        """Generate circular network plots for each module."""
        return self._plotting.module_network_plot(
            self.adata,
            n_inner=n_inner,
            n_outer=n_outer,
            n_conns=n_conns,
            mods=mods,
            outdir=outdir,
            wgcna_name=wgcna_name,
            plot_size=plot_size,
            edge_alpha=edge_alpha,
            edge_width=edge_width,
            vertex_label_cex=vertex_label_cex,
            vertex_size=vertex_size,
        )

    def generate_all_plots(self, output_dir, wgcna_name=None, formats=["pdf"]):
        """Generate all standard hdWGCNA plots."""
        return self._plotting.generate_all_plots(
            self.adata, output_dir=output_dir, wgcna_name=wgcna_name, formats=formats
        )

    # ------------------------------------------------------------------ #
    # TF Network Analysis
    # ------------------------------------------------------------------ #

    def generate_motif_data(
        self, n_tfs=100, density=0.05, seed=42, wgcna_name=None
    ):
        """Generate synthetic motif data for TF network analysis.

        Parameters
        ----------
        n_tfs : int
            Number of TFs
        density : float
            Motif-gene density
        seed : int
            Random seed
        wgcna_name : str
            Experiment name

        Returns
        -------
        HDWGCNA
            self for chaining
        """
        from .tf_network import generate_motif_data as _gm
        self.adata = _gm(
            self.adata, n_tfs=n_tfs, density=density, seed=seed,
            wgcna_name=wgcna_name
        )
        return self

    def construct_tf_network(
        self, model_params=None, nfold=5, wgcna_name=None
    ):
        """Construct directed TF-gene network using XGBoost.

        Parameters
        ----------
        model_params : dict
            XGBoost parameters
        nfold : int
            CV folds
        wgcna_name : str
            Experiment name

        Returns
        -------
        HDWGCNA
            self for chaining
        """
        from .tf_network import construct_tf_network as _ctf
        self.adata = _ctf(
            self.adata, model_params=model_params, nfold=nfold,
            wgcna_name=wgcna_name
        )
        return self

    def assign_tf_regulons(
        self, strategy="A", reg_thresh=0.01, n_tfs=10, n_genes=50,
        wgcna_name=None
    ):
        """Assign TF regulons.

        Parameters
        ----------
        strategy : str
            'A', 'B', or 'C'
        reg_thresh : float
            Gain threshold
        n_tfs : int
            Top TFs per gene (strategy A)
        n_genes : int
            Top genes per TF (strategy B)
        wgcna_name : str
            Experiment name

        Returns
        -------
        HDWGCNA
            self for chaining
        """
        from .tf_network import assign_tf_regulons as _atr
        self.adata = _atr(
            self.adata, strategy=strategy, reg_thresh=reg_thresh,
            n_tfs=n_tfs, n_genes=n_genes, wgcna_name=wgcna_name
        )
        return self

    def regulon_scores(
        self, target_type="positive", cor_thresh=0.05,
        exclude_grey_genes=True, wgcna_name=None
    ):
        """Compute regulon activity scores.

        Parameters
        ----------
        target_type : str
            'positive', 'negative', or 'both'
        cor_thresh : float
            Correlation threshold
        exclude_grey_genes : bool
            Exclude grey module genes
        wgcna_name : str
            Experiment name

        Returns
        -------
        HDWGCNA
            self for chaining
        """
        from .tf_network import regulon_scores as _rs
        self.adata = _rs(
            self.adata, target_type=target_type, cor_thresh=cor_thresh,
            exclude_grey_genes=exclude_grey_genes, wgcna_name=wgcna_name
        )
        return self

    def get_tf_target_genes(
        self, selected_tfs, depth=1, target_type="both",
        use_regulons=True, wgcna_name=None
    ):
        """Get TF target genes with depth."""
        from .tf_network import get_tf_target_genes as _gttg
        return _gttg(
            self.adata, selected_tfs=selected_tfs, depth=depth,
            target_type=target_type, use_regulons=use_regulons,
            wgcna_name=wgcna_name
        )

    def module_regulatory_network(self, tfs_only=True, wgcna_name=None):
        """Compute module-level TF regulatory network.

        Returns
        -------
        DataFrame
            Module regulatory network
        """
        from .tf_network import module_regulatory_network as _mrn
        return _mrn(self.adata, tfs_only=tfs_only, wgcna_name=wgcna_name)

    def tf_network_plot(self, selected_tfs, depth=2, wgcna_name=None,
                        save_path=None, **kwargs):
        """Plot TF-target gene network."""
        from .tf_plotting import tf_network_plot as _tnp
        return _tnp(
            self.adata, selected_tfs=selected_tfs, depth=depth,
            wgcna_name=wgcna_name, save_path=save_path, **kwargs
        )

    def regulon_bar_plot(self, selected_tf, wgcna_name=None,
                         save_path=None, **kwargs):
        """Plot regulon bar chart for a TF."""
        from .tf_plotting import regulon_bar_plot as _rbp
        return _rbp(
            self.adata, selected_tf=selected_tf,
            wgcna_name=wgcna_name, save_path=save_path, **kwargs
        )

    def module_regulatory_network_plot(
        self, feature="delta", wgcna_name=None, save_path=None, **kwargs
    ):
        """Plot module regulatory network graph."""
        from .tf_plotting import module_regulatory_network_plot as _mrnp
        return _mrnp(
            self.adata, feature=feature,
            wgcna_name=wgcna_name, save_path=save_path, **kwargs
        )

    def module_regulatory_heatmap(
        self, feature="delta", wgcna_name=None, save_path=None, **kwargs
    ):
        """Plot module regulatory heatmap."""
        from .tf_plotting import module_regulatory_heatmap as _mrh
        return _mrh(
            self.adata, feature=feature,
            wgcna_name=wgcna_name, save_path=save_path, **kwargs
        )

    # ------------------------------------------------------------------ #
    # Analysis: DMEs, Trait Correlation, Overlap
    # ------------------------------------------------------------------ #

    def find_dmes(
        self,
        group_by: str,
        group1: str,
        group2: str,
        features: str = "hMEs",
        test: str = "wilcox",
        logfc_threshold: float = 0.25,
        min_pct: float = 0.1,
        wgcna_name: str = None,
    ):
        """Find Differential Module Expression between two groups.

        Parameters
        ----------
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
        min_pct : float
        wgcna_name : str

        Returns
        -------
        DataFrame
        """
        from .analysis import find_dmes as _fd
        return _fd(
            self.adata, group_by=group_by, group1=group1, group2=group2,
            features=features, test=test, logfc_threshold=logfc_threshold,
            min_pct=min_pct, wgcna_name=wgcna_name,
        )

    def find_all_dmes(
        self,
        group_by: str,
        features: str = "hMEs",
        test: str = "wilcox",
        logfc_threshold: float = 0.25,
        min_pct: float = 0.1,
        wgcna_name: str = None,
    ):
        """Find DMEs for all pairwise comparisons within a grouping variable.

        Parameters
        ----------
        group_by : str
        features : str
        test : str
        logfc_threshold : float
        min_pct : float
        wgcna_name : str

        Returns
        -------
        dict mapping comparison name to DataFrame
        """
        from .analysis import find_all_dmes as _fad
        return _fad(
            self.adata, group_by=group_by, features=features, test=test,
            logfc_threshold=logfc_threshold, min_pct=min_pct,
            wgcna_name=wgcna_name,
        )

    def module_trait_correlation(
        self,
        trait_cols=None,
        features: str = "hMEs",
        method: str = "pearson",
        wgcna_name: str = None,
    ):
        """Compute correlation between module eigengenes and trait variables.

        Parameters
        ----------
        trait_cols : list
        features : str
        method : str
        wgcna_name : str

        Returns
        -------
        Dict with keys: cor, pval, fdr
        """
        from .analysis import module_trait_correlation as _mtc
        return _mtc(
            self.adata, trait_cols=trait_cols, features=features,
            method=method, wgcna_name=wgcna_name,
        )

    def overlap_modules_degs(
        self,
        deg_df,
        fc_cutoff: float = 0.5,
        group_col: str = "cluster",
        wgcna_name: str = None,
    ):
        """Fisher's Exact Test for overlap between DEGs and modules.

        Parameters
        ----------
        deg_df : DataFrame
        fc_cutoff : float
        group_col : str
        wgcna_name : str

        Returns
        -------
        DataFrame with overlap statistics
        """
        from .analysis import overlap_modules_degs as _omd
        return _omd(
            self.adata, deg_df=deg_df, fc_cutoff=fc_cutoff,
            group_col=group_col, wgcna_name=wgcna_name,
        )

    def module_expr_score(
        self,
        n_genes=25,
        wgcna_name=None,
    ):
        """Compute module expression scores (Seurat AddModuleScore approach).

        Parameters
        ----------
        n_genes : int or str
        wgcna_name : str

        Returns
        -------
        HDWGCNA
            self for chaining
        """
        from .analysis import module_expr_score as _mes
        self.adata = _mes(
            self.adata, n_genes=n_genes, wgcna_name=wgcna_name,
        )
        return self

    def avg_module_expr(
        self,
        n_genes=25,
        wgcna_name=None,
    ):
        """Compute average expression for each co-expression module.

        Parameters
        ----------
        n_genes : int or str
        wgcna_name : str

        Returns
        -------
        HDWGCNA
            self for chaining
        """
        from .analysis import avg_module_expr as _ame
        self.adata = _ame(
            self.adata, n_genes=n_genes, wgcna_name=wgcna_name,
        )
        return self

    # ------------------------------------------------------------------ #
    # Enrichment
    # ------------------------------------------------------------------ #

    def run_enrichr(
        self,
        gene_list,
        gene_sets: str = "GO_Biological_Process_2023",
        species: str = "human",
    ):
        """Run Enrichr enrichment analysis on a single gene list.

        Parameters
        ----------
        gene_list : list
        gene_sets : str
        species : str

        Returns
        -------
        DataFrame
        """
        from .enrichment import run_enrichr as _re
        return _re(gene_list=gene_list, gene_sets=gene_sets, species=species)

    def run_enrichr_modules(
        self,
        gene_sets=None,
        exclude_grey: bool = True,
        species: str = "human",
        wgcna_name=None,
    ):
        """Run Enrichr enrichment for each non-grey module.

        Parameters
        ----------
        gene_sets : list
        exclude_grey : bool
        species : str
        wgcna_name : str

        Returns
        -------
        dict mapping module name to enrichment DataFrame
        """
        from .enrichment import run_enrichr_modules as _rem
        return _rem(
            self.adata, gene_sets=gene_sets, exclude_grey=exclude_grey,
            species=species, wgcna_name=wgcna_name,
        )

    # ------------------------------------------------------------------ #
    # TF Network: extended functions
    # ------------------------------------------------------------------ #

    def overlap_modules_motifs(self, wgcna_name=None):
        """Test overlap between co-expression modules and TF target genes.

        Parameters
        ----------
        wgcna_name : str

        Returns
        -------
        DataFrame with module-TF overlap statistics
        """
        from .tf_network import overlap_modules_motifs as _omm
        return _omm(self.adata, wgcna_name=wgcna_name)

    def find_differential_regulons(
        self,
        barcodes1,
        barcodes2,
        test_use: str = "wilcox",
        logfc_threshold: float = 0,
        wgcna_name=None,
    ):
        """Differential regulon analysis between two groups.

        Parameters
        ----------
        barcodes1, barcodes2 : list
        test_use : str
        logfc_threshold : float
        wgcna_name : str

        Returns
        -------
        DataFrame
        """
        from .tf_network import find_differential_regulons as _fdr
        return _fdr(
            self.adata, barcodes1=barcodes1, barcodes2=barcodes2,
            test_use=test_use, logfc_threshold=logfc_threshold,
            wgcna_name=wgcna_name,
        )

    def run_enrichr_regulons(
        self,
        dbs=None,
        depth: int = 1,
        min_genes: int = 5,
        wait_time: float = 1.0,
        wgcna_name=None,
    ):
        """Run Enrichr enrichment on TF regulon target genes.

        Parameters
        ----------
        dbs : list
        depth : int
        min_genes : int
        wait_time : float
        wgcna_name : str

        Returns
        -------
        HDWGCNA
            self for chaining
        """
        from .tf_network import run_enrichr_regulons as _rer
        self.adata = _rer(
            self.adata, dbs=dbs, depth=depth, min_genes=min_genes,
            wait_time=wait_time, wgcna_name=wgcna_name,
        )
        return self

    def get_tf_network(self, wgcna_name=None):
        """Get the TF network dataframe."""
        from .tf_network import get_tf_network as _gtfn
        return _gtfn(self.adata, wgcna_name=wgcna_name)

    def get_tf_regulons(self, wgcna_name=None):
        """Get the TF regulons dataframe."""
        from .tf_network import get_tf_regulons as _gtfr
        return _gtfr(self.adata, wgcna_name=wgcna_name)

    def get_regulon_scores(self, target_type="positive", wgcna_name=None):
        """Get regulon scores."""
        from .tf_network import get_regulon_scores as _grs
        return _grs(self.adata, target_type=target_type, wgcna_name=wgcna_name)

    def get_enrichr_regulon_table(self, wgcna_name=None):
        """Get the Enrichr regulon enrichment table."""
        from .tf_network import get_enrichr_regulon_table as _gert
        return _gert(self.adata, wgcna_name=wgcna_name)

    # ------------------------------------------------------------------ #
    # Visualization: extended plotting
    # ------------------------------------------------------------------ #

    def hub_gene_network_plot(
        self,
        mods="all",
        n_hubs=6,
        n_other=3,
        sample_edges=True,
        edge_prop=0.5,
        return_graph=False,
        edge_alpha=0.25,
        vertex_label_cex=0.5,
        hub_vertex_size=4,
        other_vertex_size=1,
        wgcna_name=None,
        save_path=None,
    ):
        """Combined network plot with hub genes from multiple modules."""
        return self._plotting.hub_gene_network_plot(
            self.adata, mods=mods, n_hubs=n_hubs, n_other=n_other,
            sample_edges=sample_edges, edge_prop=edge_prop,
            return_graph=return_graph, edge_alpha=edge_alpha,
            vertex_label_cex=vertex_label_cex,
            hub_vertex_size=hub_vertex_size,
            other_vertex_size=other_vertex_size,
            wgcna_name=wgcna_name, save_path=save_path,
        )

    def module_umap_plot(
        self,
        sample_edges=True,
        edge_prop=0.2,
        label_hubs=5,
        edge_alpha=0.25,
        vertex_label_cex=0.5,
        label_genes=None,
        return_graph=False,
        keep_grey_edges=True,
        wgcna_name=None,
        save_path=None,
    ):
        """Plot module UMAP with gene network overlay."""
        return self._plotting.module_umap_plot(
            self.adata, sample_edges=sample_edges, edge_prop=edge_prop,
            label_hubs=label_hubs, edge_alpha=edge_alpha,
            vertex_label_cex=vertex_label_cex, label_genes=label_genes,
            return_graph=return_graph, keep_grey_edges=keep_grey_edges,
            wgcna_name=wgcna_name, save_path=save_path,
        )

    def compute_module_umap(
        self,
        n_hubs=10,
        exclude_grey=True,
        genes_use=None,
        n_neighbors=25,
        metric="cosine",
        spread=1.0,
        min_dist=0.4,
        supervised=False,
        random_state=42,
        wgcna_name=None,
    ):
        """Compute UMAP embedding for genes based on TOM.

        Returns
        -------
        HDWGCNA
            self for chaining
        """
        self.adata = self._plotting.compute_module_umap(
            self.adata, n_hubs=n_hubs, exclude_grey=exclude_grey,
            genes_use=genes_use, n_neighbors=n_neighbors, metric=metric,
            spread=spread, min_dist=min_dist, supervised=supervised,
            random_state=random_state, wgcna_name=wgcna_name,
        )
        return self

    def plot_dmes_volcano(
        self,
        dme_df,
        plot_labels=True,
        label_size=4,
        mod_point_size=4,
        show_cutoff=True,
        wgcna_name=None,
        xlim_range=None,
        ylim_range=None,
        save_path=None,
    ):
        """Volcano plot for DME results."""
        return self._plotting.plot_dmes_volcano(
            self.adata, dme_df=dme_df, plot_labels=plot_labels,
            label_size=label_size, mod_point_size=mod_point_size,
            show_cutoff=show_cutoff, wgcna_name=wgcna_name,
            xlim_range=xlim_range, ylim_range=ylim_range,
            save_path=save_path,
        )

    def plot_dmes_lollipop(
        self,
        dme_df,
        group_by=None,
        comparison=None,
        pvalue="p_val_adj",
        avg_log2fc_col="avg_log2FC",
        wgcna_name=None,
        save_path=None,
    ):
        """Lollipop plot for DME results."""
        return self._plotting.plot_dmes_lollipop(
            self.adata, dme_df=dme_df, group_by=group_by,
            comparison=comparison, pvalue=pvalue,
            avg_log2fc_col=avg_log2fc_col, wgcna_name=wgcna_name,
            save_path=save_path,
        )

    def plot_module_trait_correlation(
        self,
        high_color="red",
        mid_color="lightgrey",
        low_color="blue",
        label=None,
        label_symbol="stars",
        plot_max=None,
        text_size=2,
        text_color="black",
        text_digits=3,
        combine=True,
        wgcna_name=None,
        save_path=None,
    ):
        """Heatmap of module-trait correlations."""
        return self._plotting.plot_module_trait_correlation(
            self.adata, high_color=high_color, mid_color=mid_color,
            low_color=low_color, label=label, label_symbol=label_symbol,
            plot_max=plot_max, text_size=text_size, text_color=text_color,
            text_digits=text_digits, combine=combine,
            wgcna_name=wgcna_name, save_path=save_path,
        )

    def enrichr_bar_plot(
        self,
        enrichr_results,
        top_n=10,
        group_by="database",
        color="#0072B2",
        save_path=None,
    ):
        """Bar plot of Enrichr results."""
        return self._plotting.enrichr_bar_plot(
            enrichr_results=enrichr_results, top_n=top_n,
            group_by=group_by, color=color, save_path=save_path,
        )

    def enrichr_dot_plot(
        self,
        enrichr_results,
        top_n=15,
        group_by="database",
        size_col="overlap",
        save_path=None,
    ):
        """Dot plot of Enrichr results."""
        return self._plotting.enrichr_dot_plot(
            enrichr_results=enrichr_results, top_n=top_n,
            group_by=group_by, size_col=size_col, save_path=save_path,
        )

    def plot_module_preservation(
        self,
        preserv_df,
        z_thresholds=None,
        save_path=None,
    ):
        """Bar plot of module preservation statistics."""
        if z_thresholds is None:
            z_thresholds = {"high": 10, "moderate": 5, "weak": 2}
        return self._plotting.plot_module_preservation(
            preserv_df=preserv_df, z_thresholds=z_thresholds,
            save_path=save_path,
        )

    def module_dot_plot(
        self,
        features="hMEs",
        group_by="cell_type",
        exclude_grey=True,
        col_min=-2.5,
        col_max=2.5,
        dot_scale=6,
        scale_by="radius",
        rotate_x_labels=True,
        x_label_rotation=45,
        wgcna_name=None,
        save_path=None,
    ):
        """Seurat-style DotPlot for module eigengenes grouped by cell type."""
        return self._plotting.module_dot_plot(
            self.adata, features=features, group_by=group_by,
            exclude_grey=exclude_grey, col_min=col_min, col_max=col_max,
            dot_scale=dot_scale, scale_by=scale_by,
            rotate_x_labels=rotate_x_labels, x_label_rotation=x_label_rotation,
            wgcna_name=wgcna_name, save_path=save_path,
        )

    def module_radar_plot(
        self,
        group_by=None,
        barcodes=None,
        features="hMEs",
        exclude_grey=True,
        fill=True,
        draw_points=False,
        axis_label_size=4,
        grid_label_size=4,
        ncols=4,
        combine=True,
        wgcna_name=None,
        save_path=None,
    ):
        """Radar/spider plot for module eigengenes grouped by metadata."""
        return self._plotting.module_radar_plot(
            self.adata, group_by=group_by, barcodes=barcodes,
            features=features, exclude_grey=exclude_grey, fill=fill,
            draw_points=draw_points, axis_label_size=axis_label_size,
            grid_label_size=grid_label_size, ncols=ncols, combine=combine,
            wgcna_name=wgcna_name, save_path=save_path,
        )

    def module_corr_network(
        self,
        cluster_col=None,
        exclude_grey=True,
        features="hMEs",
        reduction="X_umap",
        cor_cutoff=0.2,
        label_vertices=False,
        edge_scale=5,
        vertex_size=15,
        wgcna_name=None,
        save_path=None,
    ):
        """Plot module eigengene correlation network."""
        return self._plotting.module_corr_network(
            self.adata, cluster_col=cluster_col, exclude_grey=exclude_grey,
            features=features, reduction=reduction, cor_cutoff=cor_cutoff,
            label_vertices=label_vertices, edge_scale=edge_scale,
            vertex_size=vertex_size, wgcna_name=wgcna_name,
            save_path=save_path,
        )

    def overlap_dot_plot(
        self,
        overlap_df,
        plot_var="odds_ratio",
        logscale=True,
        neglog=False,
        plot_significance=True,
        save_path=None,
    ):
        """Dot plot for module-DEG overlap results."""
        return self._plotting.overlap_dot_plot(
            overlap_df=overlap_df, plot_var=plot_var, logscale=logscale,
            neglog=neglog, plot_significance=plot_significance,
            save_path=save_path,
        )

    def overlap_bar_plot(
        self,
        overlap_df,
        plot_var="odds_ratio",
        logscale=False,
        neglog=False,
        label_size=6,
        save_path=None,
    ):
        """Bar plot for module-DEG overlap results."""
        return self._plotting.overlap_bar_plot(
            overlap_df=overlap_df, plot_var=plot_var, logscale=logscale,
            neglog=neglog, label_size=label_size, save_path=save_path,
        )

    def motif_overlap_bar_plot(
        self,
        n_tfs=10,
        module_names=None,
        wgcna_name=None,
        save_path=None,
    ):
        """Bar plot of top TFs in modules based on motif overlap."""
        return self._plotting.motif_overlap_bar_plot(
            self.adata, n_tfs=n_tfs, module_names=module_names,
            wgcna_name=wgcna_name, save_path=save_path,
        )

    def do_hub_gene_heatmap(
        self,
        n_hubs=10,
        n_cells=200,
        group_by=None,
        module_names=None,
        wgcna_name=None,
        save_path=None,
    ):
        """Heatmap of hub gene expression across cell groups."""
        return self._plotting.do_hub_gene_heatmap(
            self.adata, n_hubs=n_hubs, n_cells=n_cells, group_by=group_by,
            module_names=module_names, wgcna_name=wgcna_name,
            save_path=save_path,
        )

    def module_topology_heatmap(
        self,
        mod,
        matrix="TOM",
        order_by="kME",
        high_color=None,
        low_color="white",
        plot_max="q99",
        plot_min=0,
        wgcna_name=None,
        save_path=None,
    ):
        """Triangular heatmap of module network topology."""
        return self._plotting.module_topology_heatmap(
            self.adata, mod=mod, matrix=matrix, order_by=order_by,
            high_color=high_color, low_color=low_color, plot_max=plot_max,
            plot_min=plot_min, wgcna_name=wgcna_name, save_path=save_path,
        )

    def module_topology_barplot(
        self,
        mod,
        features="kME",
        plot_color=None,
        alpha=True,
        wgcna_name=None,
        save_path=None,
    ):
        """Ranked barplot of intramodular connectivity."""
        return self._plotting.module_topology_barplot(
            self.adata, mod=mod, features=features, plot_color=plot_color,
            alpha=alpha, wgcna_name=wgcna_name, save_path=save_path,
        )

    def plot_module_preservation_lollipop(
        self,
        preservation_name,
        features=None,
        fdr=True,
        wgcna_name=None,
        save_path=None,
    ):
        """Lollipop plot for module preservation statistics."""
        return self._plotting.plot_module_preservation_lollipop(
            self.adata, preservation_name=preservation_name, features=features,
            fdr=fdr, wgcna_name=wgcna_name, save_path=save_path,
        )

    # ------------------------------------------------------------------ #
    # TF Visualization: extended
    # ------------------------------------------------------------------ #

    def plot_differential_regulons(
        self,
        dregs,
        n_label=10,
        logfc_thresh=0.1,
        lm=True,
        figsize=(8, 8),
        wgcna_name=None,
        save_path=None,
    ):
        """Scatter plot of differential regulon results."""
        from .tf_plotting import plot_differential_regulons as _pdr
        return _pdr(
            self.adata, dregs=dregs, n_label=n_label,
            logfc_thresh=logfc_thresh, lm=lm, figsize=figsize,
            wgcna_name=wgcna_name, save_path=save_path,
        )
