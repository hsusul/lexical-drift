from __future__ import annotations

import importlib


def test_imports_do_not_require_torch_at_module_import_time() -> None:
    modules = [
        "lexical_drift.models.temporal_encoder",
        "lexical_drift.losses.infonce",
        "lexical_drift.train.e2e_temporal",
        "lexical_drift.train.contrastive_temporal",
        "lexical_drift.train.multitask_temporal",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
