from __future__ import annotations

import importlib.util

import pytest

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason='requires torch (install with: pip install -e ".[torch]")',
)
