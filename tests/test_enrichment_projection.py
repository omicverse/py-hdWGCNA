"""Tests for enrichment and projection modules."""
import pytest
import numpy as np
import pandas as pd


class TestEnrichment:
    """Test Enrichr integration."""

    def test_run_enrichr_input_validation(self):
        from py_hdWGCNA.enrichment import run_enrichr
        genes = ['TP53', 'BRCA1', 'EGFR', 'MYC', 'AKT1']
        df = run_enrichr(genes, gene_sets='GO_Biological_Process_2023')
        assert df is not None
        assert isinstance(df, pd.DataFrame)

    def test_run_enrichr_empty_list(self):
        from py_hdWGCNA.enrichment import run_enrichr
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = run_enrichr([], gene_sets='GO_Biological_Process_2023')
            assert len(df) == 0


class TestProjection:
    """Test module projection."""

    def test_project_modules_requires_shared_genes(self, test_adata, wgcna_name):
        from py_hdWGCNA import (
            setup_for_wgcna, test_soft_powers,
            construct_network, module_eigengenes, project_modules,
        )
        src = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        src = test_soft_powers(src, power_range=range(2, 8), wgcna_name=wgcna_name)
        src = construct_network(src, power=6, wgcna_name=wgcna_name, minModuleSize=10)
        src = module_eigengenes(src, wgcna_name=wgcna_name)
        import anndata
        tgt = anndata.AnnData(
            X=np.random.default_rng(99).normal(0, 1, size=(50, 20)),
            obs=pd.DataFrame(index=[f'C{i}' for i in range(50)]),
            var=pd.DataFrame(index=[f'G{i}' for i in range(20)])
        )
        with pytest.raises(ValueError, match="shared genes"):
            project_modules(src, tgt, wgcna_name=wgcna_name)
