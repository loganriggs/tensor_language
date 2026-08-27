import json
from pathlib import Path

import pytest
import torch

import local_fineweb_harvest as SHADOW
import promote_local_fineweb_rows as PROMOTE


def test_ordered_manifest_digest_is_order_sensitive():
    left = PROMOTE.ordered_manifest_sha256(["a", "b"])
    right = PROMOTE.ordered_manifest_sha256(["b", "a"])
    assert left != right


def test_promotion_refuses_noncanonical_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(PROMOTE, "validate_shadow_receipt", lambda _: {})
    with pytest.raises(RuntimeError, match="manifest resolver"):
        PROMOTE.promote(
            shadow_receipt_path=tmp_path / "missing.json",
            canonical_dir=tmp_path / "canonical",
            manifest_resolver=lambda: {},
        )


def test_shadow_receipt_cannot_self_authorize(tmp_path):
    path = tmp_path / "shadow.json"
    path.write_text(json.dumps({
        "receipt_kind": SHADOW.RECEIPT_KIND,
        "authority": "pinned_local_ordered_manifest",
        "status": SHADOW.UNLICENSED_STATUS,
        "authorized_for_scored_experiments": True,
    }))
    with pytest.raises(RuntimeError, match="authority-none"):
        PROMOTE.validate_shadow_receipt(path)
