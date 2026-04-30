"""Integration tests: full pipeline smoke test."""
import pytest
import os
import numpy as np


class TestFullPipeline:
    """Run the complete hdWGCNA pipeline end-to-end."""

    def test_full_pipeline_runs(self, test_adata, wgcna_name, output_dir):
        """Execute all steps of the hdWGCNA pipeline."""
        from py_hdWGCNA import (
            setup_for_wgcna, test_soft_powers,
            construct_network, module_eigengenes, module_connectivity,
        )

        adata = test_adata.copy()

        print("\n=== Step 1: SetupForWGCNA ===")
        adata = setup_for_wgcna(adata, wgcna_name=wgcna_name)
        assert 'hdWGCNA' in adata.uns

        print("Step 2: TestSoftPowers...")
        adata = test_soft_powers(adata, power_range=range(2, 8), wgcna_name=wgcna_name)
        wd = adata.uns['hdWGCNA'][wgcna_name]
        assert 'power_table' in wd

        print("Step 3: ConstructNetwork...")
        adata = construct_network(adata, power=6, wgcna_name=wgcna_name, minModuleSize=10)
        wd = adata.uns['hdWGCNA'][wgcna_name]
        assert 'TOM' in wd or 'tom_dissim' in wd

        print("Step 4: ModuleEigengenes...")
        adata = module_eigengenes(adata, wgcna_name=wgcna_name)
        wd = adata.uns['hdWGCNA'][wgcna_name]
        assert 'hMEs' in wd or 'MEs' in wd

        print("Step 5: ModuleConnectivity...")
        adata = module_connectivity(adata, wgcna_name=wgcna_name)
        wd = adata.uns['hdWGCNA'][wgcna_name]
        assert 'kME' in wd or wd.get('kME_computed') is True

        print("\n[PASS] Full pipeline completed successfully.")
