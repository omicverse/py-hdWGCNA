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
        minModuleSize: int = 50,
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
        wgcna_name: str = None,
    ):
        """Compute module eigengenes (MEs) in single cells.

        Parameters
        ----------
        group_by_vars : str or list
            Variable(s) to harmonize by
        harmonize : bool
            Apply Harmony correction
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
