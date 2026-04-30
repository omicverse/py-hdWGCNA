"""Tests for module detection and eigengenes."""

import numpy as np
import pandas as pd


class TestModuleEigengenes:
    """Test module eigengene computation."""

    def test_module_eigengenes_basic(self, test_adata, wgcna_name):
        from py_hdWGCNA import (
            setup_for_wgcna,
            test_soft_powers,
            construct_network,
            module_eigengenes,
        )

        adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        adata = test_soft_powers(adata, power_range=range(2, 8), wgcna_name=wgcna_name)
        adata = construct_network(
            adata, power=6, wgcna_name=wgcna_name, minModuleSize=10
        )
        adata = module_eigengenes(adata, wgcna_name=wgcna_name)
        wd = adata.uns["hdWGCNA"][wgcna_name]
        assert "hMEs" in wd or "MEs" in wd

    def test_hme_storage(self, test_adata, wgcna_name):
        from py_hdWGCNA import (
            setup_for_wgcna,
            test_soft_powers,
            construct_network,
            module_eigengenes,
        )

        adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        adata = test_soft_powers(adata, power_range=range(2, 8), wgcna_name=wgcna_name)
        adata = construct_network(
            adata, power=6, wgcna_name=wgcna_name, minModuleSize=10
        )
        adata = module_eigengenes(adata, wgcna_name=wgcna_name)
        wd = adata.uns["hdWGCNA"][wgcna_name]
        assert "hMEs" in wd
        hMEs = wd["hMEs"]
        if isinstance(hMEs, np.ndarray):
            assert hMEs.shape[0] > 0
        elif isinstance(hMEs, pd.DataFrame):
            assert hMEs.shape[0] > 0


class TestModuleConnectivity:
    """Test kME computation."""

    def test_kme_computation(self, test_adata, wgcna_name):
        from py_hdWGCNA import (
            setup_for_wgcna,
            test_soft_powers,
            construct_network,
            module_eigengenes,
            module_connectivity,
        )

        adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        adata = test_soft_powers(adata, power_range=range(2, 8), wgcna_name=wgcna_name)
        adata = construct_network(
            adata, power=6, wgcna_name=wgcna_name, minModuleSize=10
        )
        adata = module_eigengenes(adata, wgcna_name=wgcna_name)
        adata = module_connectivity(adata, wgcna_name=wgcna_name)
        wd = adata.uns["hdWGCNA"][wgcna_name]
        assert "kME" in wd or wd.get("kME_computed") is True
        modules_df = wd.get("modules_df")
        if modules_df is not None:
            kme_cols = [c for c in modules_df.columns if "kME" in c]
            assert len(kme_cols) > 0
