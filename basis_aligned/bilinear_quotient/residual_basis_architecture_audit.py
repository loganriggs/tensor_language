#!/usr/bin/env python3
"""Static source-closure audit for the global residual O(D) gauge claim."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "residual_basis_architecture_contract.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(contract_path=CONTRACT, root=ROOT):
    contract = json.loads(Path(contract_path).read_text())
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported architecture contract")
    texts = {}
    for source in contract["sources"]:
        path = Path(root)/source["path"]
        if sha(path) != source["sha256"]:
            raise ValueError(f"source hash mismatch: {source['path']}")
        text = path.read_text()
        ast.parse(text)
        texts[source["path"]] = text
    runtime = texts["jacclust/tt_model.py"]
    reference = texts[
        "basis_aligned/bilinear_quotient/bilin18_reference_forward.py"]
    for fragment in contract["required_runtime_fragments"]:
        if fragment not in runtime:
            raise ValueError(f"runtime obligation missing: {fragment}")
    for fragment in contract["required_reference_fragments"]:
        if fragment not in reference:
            raise ValueError(f"reference obligation missing: {fragment}")
    claims = contract["claims"]
    if claims != {"exact_over_real_arithmetic": True,
                  "exact_checkpoint_bit_replay_after_rotation": False,
                  "finite_precision_logit_identity_certified": False,
                  "checkpoint_anchor_generic_stratum_certified": False,
                  "global_quotient_price_certified": False}:
        raise ValueError("claim boundary changed")
    return {"contract_id": contract["contract_id"], "sources_verified": 2,
            "runtime_obligations_verified": len(contract["required_runtime_fragments"]),
            "reference_obligations_verified": len(contract["required_reference_fragments"]),
            "claims": claims}
