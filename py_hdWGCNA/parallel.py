"""
Parallelization utilities for py-hdWGCNA.

Provides CPU-based parallelization using concurrent.futures for
compute-intensive operations like soft power testing, TF network
construction, and matrix computations.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def _get_max_workers(n_threads: int | None = None) -> int:
    """Resolve the number of worker threads/processes.

    Parameters
    ----------
    n_threads : int or None
        Requested number of threads. None or -1 means auto-detect
        based on CPU count. Values are clamped to [1, cpu_count].

    Returns
    -------
    int
        Resolved worker count
    """
    cpu_count = os.cpu_count() or 4
    if n_threads is None or n_threads <= 0:
        # Use at most cpu_count, but cap at 8 to avoid oversubscription
        return min(cpu_count, 8)
    return min(n_threads, cpu_count)


def parallel_map(func, items, n_workers: int | None = None):
    """Apply func to each item in items using a process pool.

    Parameters
    ----------
    func : callable
        Function to apply. Must be picklable (top-level or lambda).
    items : iterable
        Items to process.
    n_workers : int or None
        Number of worker processes. None = auto.

    Returns
    -------
    list
        Results in the same order as items.
    """
    n_workers = _get_max_workers(n_workers)
    if n_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    results = [None] * len(items)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return results
