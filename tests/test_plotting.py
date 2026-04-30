"""Tests for plotting functions."""
import pytest
import os


class TestBasePlots:
    """Test base visualization functions."""

    @pytest.fixture(autouse=True)
    def _setup(self, test_adata, wgcna_name):
        from py_hdWGCNA import (
            setup_for_wgcna, test_soft_powers,
            construct_network, module_eigengenes, module_connectivity,
        )
        self.adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        try:
            self.adata = test_soft_powers(self.adata, power_range=range(2, 8), wgcna_name=wgcna_name)
            self.adata = construct_network(self.adata, power=6, wgcna_name=wgcna_name, minModuleSize=10)
            self.adata = module_eigengenes(self.adata, wgcna_name=wgcna_name)
            self.adata = module_connectivity(self.adata, wgcna_name=wgcna_name)
        except Exception:
            pass
        self.wgcna_name = wgcna_name

    def test_plot_soft_powers(self, output_dir):
        from py_hdWGCNA.plotting import plot_soft_powers
        wd = self.adata.uns['hdWGCNA'][self.wgcna_name]
        if 'power_table' not in wd:
            pytest.skip("Power table not computed")
        save_path = os.path.join(output_dir, 'soft_powers.pdf')
        fig = plot_soft_powers(self.adata, wgcna_name=self.wgcna_name, save_path=save_path)
        assert fig is not None

    def test_plot_kmes(self, output_dir):
        from py_hdWGCNA.plotting import plot_kmes
        wd = self.adata.uns['hdWGCNA'][self.wgcna_name]
        if 'modules_df' not in wd:
            pytest.skip("Modules not computed")
        modules_df = wd['modules_df']
        kme_cols = [c for c in modules_df.columns if 'kME' in c]
        if len(kme_cols) == 0:
            pytest.skip("kME not computed")
        save_path = os.path.join(output_dir, 'kmes.pdf')
        fig = plot_kmes(self.adata, wgcna_name=self.wgcna_name, save_path=save_path)
        assert fig is not None

    def test_module_correlogram(self, output_dir):
        from py_hdWGCNA.plotting import module_correlogram
        wd = self.adata.uns['hdWGCNA'][self.wgcna_name]
        if 'hMEs' not in wd and 'MEs' not in wd:
            pytest.skip("Module eigengenes not computed")
        save_path = os.path.join(output_dir, 'correlogram.pdf')
        fig = module_correlogram(self.adata, wgcna_name=self.wgcna_name, save_path=save_path)
        assert fig is not None
