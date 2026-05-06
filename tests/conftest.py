"""
Shared fixtures for py-hdWGCNA test suite.
"""

import sys
import os
import pytest

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session")
def test_adata():
    """
    Load test AnnData from h5ad file with pre-computed hdWGCNA results.
    Returns an adata object ready for testing all functions.
    """
    import scanpy as sc

    h5ad_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "examples", "test_seurat.h5ad")
    )

    if not os.path.exists(h5ad_path):
        pytest.skip(f"Test data not found: {h5ad_path}")

    try:
        adata = sc.read_h5ad(h5ad_path)
    except Exception as e:
        pytest.skip(f"Could not load test h5ad file: {e}")

    if adata.X is None:
        pytest.skip("No expression matrix in h5ad file.")

    return adata


@pytest.fixture(scope="session")
def wgcna_name():
    """Return the default WGCNA experiment name."""
    return "HDWGCNA"


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory):
    """Create temporary output directory for tests."""
    return str(tmp_path_factory.mktemp("hdwgcna_test"))
