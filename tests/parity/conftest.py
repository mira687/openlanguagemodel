"""Parity-suite fixtures.

The suite must be importable as ``tests.parity`` from the repository root, and
must run identically whether invoked as ``pytest tests/parity`` or as part of a
full ``pytest`` run.
"""

from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_cpu():
    """Pin threading and algorithm choice around every parity test."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)
