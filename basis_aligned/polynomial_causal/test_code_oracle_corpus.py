import hashlib
import json
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent


def test_legacy_v1_code_oracle_corpus_integrity_is_retained():
    """V1 is preserved for audit only; SHIP_CONTENT_OOD_ORACLE_SPEC bars scoring it."""
    payload = torch.load(HERE / "code_oracle_corpus.pt", weights_only=False)
    manifest = json.loads((HERE / "code_oracle_corpus_manifest.json").read_text())
    rows = payload["rows"]
    assert payload["manifest"] == manifest
    assert manifest["schema_version"] == 1
    assert tuple(rows.shape) == (480, 257)
    assert rows.dtype == torch.long
    assert manifest["splits"] == {
        "basis": [0, 96], "discovery": [96, 288], "heldout": [288, 480]
    }
    assert manifest["tensor_raw_sha256"] == hashlib.sha256(
        rows.numpy().tobytes(order="C")
    ).hexdigest()
    assert manifest["construction_script_sha256"] == hashlib.sha256(
        (HERE / "freeze_code_oracle_corpus.py").read_bytes()
    ).hexdigest()
    assert sum(row["used_tokens"] for row in manifest["files"]) == rows.numel()
    assert len({row["path"] for row in manifest["files"]}) == len(manifest["files"])
