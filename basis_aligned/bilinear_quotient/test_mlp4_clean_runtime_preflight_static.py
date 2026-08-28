import ast
import hashlib
import json

from pathlib import Path


HERE = Path(__file__).resolve().parent
PROTOCOL = HERE/"mlp4_clean_runtime_preflight_protocol.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_preflight_is_synthetic_only_and_source_pinned():
    protocol = json.loads(PROTOCOL.read_text())
    for filename, expected in protocol["pinned_sources"].items():
        assert sha(HERE/filename) == expected
    source = (HERE/"mlp4_clean_runtime_preflight.py").read_text()
    ast.parse(source)
    assert "torch.arange" in source and "reference_forward" in source
    for forbidden in ("fineweb", "ROWS", "validation_rows", "fit_rows",
                      "census", "datasets", "torch.load"):
        assert forbidden not in source
    assert protocol["synthetic_input"]["token_ids"] == "arange(32)"
    assert protocol["resources"]["hard_abort_peak_gib"] <= 5
    assert protocol["resources"]["hard_abort_temperature_c"] <= 82
    assert not protocol["permissions"]["validation_rows_may_be_opened"]
    assert not protocol["permissions"]["any_natural_rows_may_be_opened"]


def test_independent_reference_is_data_free_and_does_not_call_model_blocks():
    source = (HERE/"bilin18_reference_forward.py").read_text()
    ast.parse(source)
    for forbidden in ("torch.load", "huggingface", "dataset", "fineweb",
                      "block(", "attention(", "mlp("):
        assert forbidden not in source
    assert "mlp.Down(mlp.Left(z)*mlp.Right(z))+mlp.Down_bias" in source
    assert "score1*score2" in source and "masked_fill" in source
