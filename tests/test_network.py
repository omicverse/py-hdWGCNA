"""Tests for network construction module."""
import pytest
import numpy as np
import pandas as pd


class TestSoftPowers:
    """Test soft power threshold selection."""

    def test_soft_powers_basic(self, test_adata, wgcna_name):
        from py_hdWGCNA import setup_for_wgcna, test_soft_powers
        adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        adata = test_soft_powers(adata, power_range=range(2, 10), wgcna_name=wgcna_name)
        wd = adata.uns['hdWGCNA'][wgcna_name]
        assert 'power_table' in wd
        pt = wd['power_table']
        assert isinstance(pt, pd.DataFrame)
        assert 'Power' in pt.columns
        assert 'SFT.R.sq' in pt.columns

    def test_power_table_storage(self, test_adata, wgcna_name):
        from py_hdWGCNA import setup_for_wgcna, test_soft_powers
        adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        adata = test_soft_powers(adata, power_range=[2, 4, 6, 8], wgcna_name=wgcna_name)
        wd = adata.uns['hdWGCNA'][wgcna_name]
        assert 'power_table' in wd

    def test_select_power(self, test_adata, wgcna_name):
        from py_hdWGCNA import setup_for_wgcna, test_soft_powers
        adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        adata = test_soft_powers(adata, power_range=range(2, 12), wgcna_name=wgcna_name)
        wd = adata.uns['hdWGCNA'][wgcna_name]
        pt = wd['power_table']
        valid = pt[pt['SFT.R.sq'] >= 0.8]
        if len(valid) > 0:
            selected_power = int(valid.iloc[0]['Power'])
            assert selected_power >= 2


class TestConstructNetwork:
    """Test network construction and TOM computation."""

    def test_construct_network_basic(self, test_adata, wgcna_name):
        from py_hdWGCNA import setup_for_wgcna, test_soft_powers, construct_network
        adata = setup_for_wgcna(test_adata.copy(), wgcna_name=wgcna_name)
        adata = test_soft_powers(adata, power_range=range(2, 8), wgcna_name=wgcna_name)
        adata = construct_network(adata, power=6, wgcna_name=wgcna_name, minModuleSize=10)
        assert adata is not None
        wd = adata.uns['hdWGCNA'][wgcna_name]
        assert 'TOM' in wd or 'tom_dissim' in wd
        assert 'modules_df' in wd
