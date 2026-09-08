import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run_temporal_iswas_selected_writer_block10_direct_residual_add_remove_v2_scoped_logit_audit as target


def test_dryrun_is_authority_bound_and_model_free():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["comparison_scope"] == "iswas_query_answer_and_foil"
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 14


def test_scoped_comparator_ignores_out_of_footprint_logits(monkeypatch):
    endpoints = [({}, None, {"iswas_position": 2}),
                 ({}, None, {"iswas_position": 1})]
    loc = target.v1.factorial.atlas.mediation.parent.loc
    monkeypatch.setattr(loc.parent, "endpoint_bank", lambda _rows: (endpoints, {}))
    monkeypatch.setattr(loc.original, "_single", lambda text: {" is": 3, " was": 4}[text])
    authority = type("Authority", (), {"build_rows": staticmethod(lambda: [{}, {}])})
    compare = target.selected_logit_comparator(authority)
    left = torch.zeros(2, 3, 6)
    right = left.clone()
    right[:, 2, 5] = 1000
    assert compare(left, right) == 0.0
    right[0, 1, 3] = 0.25
    assert compare(left, right) == 0.25


def test_v1_failure_is_preserved_and_v2_price_is_unchanged():
    prior = json.loads(target.PRIOR.read_text())
    source = json.loads(target.V1_RESULT.read_text())
    assert source["terminal"] == "arithmetic_or_hook_failure"
    assert tuple(source["predictions"].values()) == (True, False, False, False, False)
    assert prior["price"] == target.PRICE == target.v1.PRICE
    assert {name: target.sha(path) for name, path in target.FILES.items()} == target.EXPECTED
