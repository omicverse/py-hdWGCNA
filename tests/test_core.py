"""Tests for hdWGCNA core module (SetupForWGCNA)."""


class TestSetupForWGCNA:
    """Test gene selection for WGCNA."""

    def test_setup_for_wgcna_basic(self, test_adata, wgcna_name, output_dir):
        from py_hdWGCNA import setup_for_wgcna

        adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        assert "hdWGCNA" in adata.uns
        assert wgcna_name in adata.uns["hdWGCNA"]
        wd = adata.uns["hdWGCNA"][wgcna_name]
        assert "dat_expr" in wd
        assert "dat_expr_genes" in wd
        assert wd["setup_complete"] is True
        assert len(wd["genes_use"]) > 0

    def test_variable_gene_selection(self, test_adata, wgcna_name):
        from py_hdWGCNA import setup_for_wgcna

        adata = setup_for_wgcna(
            test_adata.copy(),
            gene_select="variable",
            n_genes=200,
            wgcna_name=wgcna_name,
        )
        wd = adata.uns["hdWGCNA"][wgcna_name]
        assert wd["gene_select_method"] == "variable"
        assert len(wd["genes_use"]) > 0
        assert wd["n_genes"] <= 200

    def test_fraction_gene_selection(self, test_adata, wgcna_name):
        from py_hdWGCNA import setup_for_wgcna

        adata = setup_for_wgcna(
            test_adata.copy(),
            gene_select="fraction",
            fraction=0.05,
            wgcna_name=wgcna_name,
        )
        wd = adata.uns["hdWGCNA"][wgcna_name]
        assert wd["gene_select_method"] == "fraction"
        assert len(wd["genes_use"]) > 0
        assert wd["dat_expr"].shape[0] == len(wd["dat_expr_genes"])

    def test_custom_gene_selection(self, test_adata, wgcna_name):
        from py_hdWGCNA import setup_for_wgcna

        custom_genes = list(test_adata.var_names[:50])
        adata = setup_for_wgcna(
            test_adata.copy(),
            gene_select="custom",
            genes_use=custom_genes,
            wgcna_name=wgcna_name,
        )
        wd = adata.uns["hdWGCNA"][wgcna_name]
        assert wd["gene_select_method"] == "custom"
        assert len(wd["genes_use"]) > 0
