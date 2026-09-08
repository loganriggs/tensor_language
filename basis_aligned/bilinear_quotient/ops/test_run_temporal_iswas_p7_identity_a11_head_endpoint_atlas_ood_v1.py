import json
import subprocess
import sys

import torch

import run_temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1 as target


def test_unbound_runner_exits_model_free():
    completed = subprocess.run([sys.executable, target.__file__], check=True,
        capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["status"] == "awaiting_binding"
    assert payload["authority_ok"] is True
    assert payload["gpu_accessed"] is payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 22


def test_head_slice_replacement_is_exact_and_scoped():
    native = torch.zeros(2, 3, 18)
    writer = torch.arange(108, dtype=torch.float32).reshape(2, 3, 18)
    changed = target.replace_head_slices(native, writer, (3, 8), [[0, 2], [1]], n_heads=9)
    for row, positions in enumerate(([0, 2], [1])):
        for position in range(3):
            for head in range(9):
                sl = slice(2 * head, 2 * head + 2)
                expected = writer if position in positions and head in (3, 8) else native
                assert torch.equal(changed[row, position, sl], expected[row, position, sl])


def test_invalid_head_geometry_fails_closed():
    x = torch.zeros(1, 1, 18)
    for heads in ((9,), (3, 3)):
        try:
            target.replace_head_slices(x, x, heads, [[0]], n_heads=9)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid head set accepted")
