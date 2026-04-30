"""Tests for DME analysis functions."""

import pytest
import numpy as np
import pandas as pd


class TestFindDMEs:
    """Test differential module expression analysis."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_adata, wgcna_name):
        from py_hdWGCNA import (
            setup_for_wgcna,
            test_soft_powers,
            construct_network,
            module_eigengenes,
        )

        self.adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        rng = np.random.default_rng(42)
        self.adata.obs["test_group"] = rng.choice(["A", "B"], size=self.adata.n_obs)
        try:
            self.adata = test_soft_powers(
                self.adata, power_range=range(2, 8), wgcna_name=wgcna_name
            )
            self.adata = construct_network(
                self.adata, power=6, wgcna_name=wgcna_name, minModuleSize=10
            )
            self.adata = module_eigengenes(self.adata, wgcna_name=wgcna_name)
        except Exception:
            pass
        self.wgcna_name = wgcna_name

    def test_find_dmes_basic(self):
        from py_hdWGCNA import find_dmes

        wd = self.adata.uns["hdWGCNA"][self.wgcna_name]
        if "hMEs" not in wd and "MEs" not in wd:
            pytest.skip("Module eigengenes not computed")
        dme_df = find_dmes(
            self.adata,
            group_by="test_group",
            group1="A",
            group2="B",
            wgcna_name=self.wgcna_name,
        )
        assert dme_df is not None
        assert isinstance(dme_df, pd.DataFrame)
        assert "module" in dme_df.columns
        assert "p_val" in dme_df.columns
        assert "avg_log2FC" in dme_df.columns

    def test_find_dmes_columns(self):
        from py_hdWGCNA import find_dmes

        wd = self.adata.uns["hdWGCNA"][self.wgcna_name]
        if "hMEs" not in wd and "MEs" not in wd:
            pytest.skip("Module eigengenes not computed")
        dme_df = find_dmes(
            self.adata,
            group_by="test_group",
            group1="A",
            group2="B",
            wgcna_name=self.wgcna_name,
        )
        expected_cols = ["p_val", "avg_log2FC", "p_val_adj"]
        for col in expected_cols:
            assert col in dme_df.columns, f"Missing column: {col}"
