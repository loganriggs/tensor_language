#!/usr/bin/env python3
"""Focused CPU receipt for the shared exact-batch and predicate preflight."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OPS = ROOT / "basis_aligned/bilinear_quotient/ops"
sys.path.insert(0, str(OPS))
import circuit_exactness_preflight as preflight
import gate


OUT = HERE / "SHARED_EXACTNESS_PREFLIGHT_V1_RESULT.json"
V1 = HERE / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_RESULT.json"
V2 = HERE / "SUCCESSOR_POINTER_PREFIX_INTERACTION_V2_RESULT.json"


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_v = torch.nn.Linear(4, 4, bias=False)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("lambdas", torch.tensor([0.75, 0.25]))
        self.attn = Attention()


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = torch.nn.Embedding(13, 4)
        self.h = torch.nn.ModuleList([Block()])


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    torch.manual_seed(7)
    model = Model()
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    state = F.rms_norm(model.transformer.wte(tokens), (4,))
    first = state
    block = model.transformer.h[0]
    state = block.lambdas[0] * state + block.lambdas[1] * first
    expected = block.attn.c_v(F.rms_norm(state, (4,))).view(2, 3, 2, 2)
    actual = preflight.exact_static_first_value(
        model, {"n_embd": 4, "n_head": 2}, tokens, F)
    exact_error = float((actual - expected).detach().abs().max())
    try:
        preflight.exact_static_first_value(
            model, {"n_embd": 4, "n_head": 2}, tokens[:, 0], F)
    except ValueError:
        selected_vector_refused = True
    else:
        selected_vector_refused = False

    fixture = """import json
def main():
    json.dump(dict(pred_a_x=True, pred_b_y=True, pred_c_z=True), open('/dev/null', 'w'))
main()
"""
    keys = preflight.static_prediction_keys(fixture)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(fixture)
        fixture_path = handle.name
    try:
        gate_findings = gate.gate(fixture_path)
    finally:
        os.unlink(fixture_path)

    fast = subprocess.run(
        [sys.executable, str(OPS / "test_fast.py")],
        text=True, capture_output=True, timeout=30,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "BQLIB_NO_MODEL": "1"},
    )
    v1 = json.loads(V1.read_text())
    v2 = json.loads(V2.read_text())
    passed = (
        exact_error == 0.0 and selected_vector_refused
        and keys == ["pred_a_x", "pred_b_y", "pred_c_z"]
        and not gate_findings and fast.returncode == 0
        and v1["exactness"]["maximum_self_logit_absolute_error"] > 1e-5
        and v2["exactness"]["maximum_self_logit_absolute_error"] == 0.0
    )
    result = {
        "schema": "shared_exactness_preflight_v1_result",
        "passed": passed,
        "same_batch_static_value_max_absolute_error": exact_error,
        "selected_token_vector_refused": selected_vector_refused,
        "static_predicate_keys": keys,
        "static_gate_findings": gate_findings,
        "shared_fast_suite_exit_code": fast.returncode,
        "shared_fast_suite_terminal": fast.stdout.strip().splitlines()[-1],
        "historical_reproduction": {
            "v1_singleton_batch_self_error": v1["exactness"]["maximum_self_logit_absolute_error"],
            "v2_native_batch_self_error": v2["exactness"]["maximum_self_logit_absolute_error"],
            "frozen_bar": 1e-5,
        },
        "files": {
            str(path.relative_to(ROOT)): sha(path) for path in (
                OPS / "circuit_exactness_preflight.py", OPS / "gate.py",
                OPS / "test_fast.py", V1, V2,
            )
        },
        "model_loaded": False,
        "gpu_accessed": False,
    }
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
