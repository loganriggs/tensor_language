import importlib.util
import json
from pathlib import Path

import pytest
import torch


PATH = Path(__file__).with_name("prepare_fineweb_oracle_rows.py")
SPEC = importlib.util.spec_from_file_location("prepare_fineweb_oracle_rows", PATH)
PREP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PREP)


def test_receipt_content_addresses_every_required_row_set(monkeypatch, tmp_path):
    monkeypatch.setattr(PREP.rowcache, "CACHE", str(tmp_path / "cache"))
    rows = {}
    for n, skip in PREP.SPECS:
        tensor = torch.arange(n * PREP.rowcache.T_LEN, dtype=torch.long).view(
            n, PREP.rowcache.T_LEN
        ) + skip
        rows[(n, skip)] = tensor
        path = Path(PREP.rowcache._path(n, skip))
        path.parent.mkdir(parents=True, exist_ok=True)
        PREP.rowcache._save_atomic(tensor, str(path))
    receipt = PREP.build_receipt(rows)
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))

    loaded_receipt, loaded = PREP.validate_receipt(path)
    assert loaded_receipt == receipt
    for spec in PREP.SPECS:
        assert torch.equal(loaded[spec], rows[spec])

    target = PREP.SPECS[-1]
    tampered = rows[target].clone()
    tampered[0, 0] += 1
    PREP.rowcache._save_atomic(tampered, PREP.rowcache._path(*target))
    with pytest.raises(RuntimeError, match="hash mismatch"):
        PREP.validate_receipt(path)


def test_receipt_requires_real_stream_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(PREP.rowcache, "CACHE", str(tmp_path / "cache"))
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps({"real_stream_bit_identity_gate": {"passed": False}}))
    with pytest.raises(RuntimeError, match="identity gate"):
        PREP.validate_receipt(path)
