#!/usr/bin/env python3
# BQLANE: cpu
"""Independent model-free review of the exact R592 topology amendment."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import subprocess


COMMIT = "0f33456126fa2ea5233798d937e61f7dd6a0ea93"
ROOT = Path(__file__).resolve().parents[3]
AMENDMENT = "basis_aligned/polynomial_causal/INDUCTION_CENTERED_FIXED_GEOMETRY_RUNG592_LOGIT_TOPOLOGY_AMENDMENT.md"
FACADE = "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py"
AMENDMENT_SHA256 = "15219749dd1d696e52c3129052cadce6758b7186390303eace216d98c953188e"
FACADE_SHA256 = "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c"


def blob(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{COMMIT}:{path}"], cwd=ROOT)


def literal_constants(source: bytes) -> dict[str, object]:
    output: dict[str, object] = {}
    for node in ast.parse(source).body:
        target = (
            node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else node.target if isinstance(node, ast.AnnAssign) else None
        )
        if isinstance(target, ast.Name):
            try:
                output[target.id] = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                pass
    return output


def test_exact_commit_contains_only_exact_amendment() -> None:
    assert subprocess.check_output(["git", "rev-parse", COMMIT], cwd=ROOT, text=True).strip() == COMMIT
    changed = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", COMMIT],
        cwd=ROOT, text=True,
    ).splitlines()
    assert changed == [AMENDMENT]
    assert hashlib.sha256(blob(AMENDMENT)).hexdigest() == AMENDMENT_SHA256


def test_pinned_facade_topology_is_50304_and_forward_is_unsliced() -> None:
    source = blob(FACADE)
    assert hashlib.sha256(source).hexdigest() == FACADE_SHA256
    values = literal_constants(source)
    assert values["TOKENIZER_VOCAB"] == 50_257
    assert values["LOGIT_VOCAB"] == 50_304
    text = source.decode()
    assert '"vocab_size": LOGIT_VOCAB' in text
    body = text[text.index("def forward_with_dispatch"):]
    body = body[:body.index("\ndef ", 1)]
    assert "expected_vocab = LOGIT_VOCAB if require_production else model.config.vocab_size" in body
    assert "return logits" in body
    assert "logits[:" not in body and "logits[...," not in body


def test_every_corrected_data_byte_formula() -> None:
    vocab = 50_304
    assert vocab * 4 == 201_216
    assert 32 * vocab * 4 == 6_438_912
    assert 16 * vocab * 4 == 3_219_456

    fit_logits = 3_744 * 4 * vocab * 4
    select_logits = 1_872 * 4 * vocab * 4
    assert fit_logits == 3_013_410_816
    assert select_logits == 1_506_705_408
    assert fit_logits + select_logits == 4_520_116_224

    fit_hooks = 3_744 * 4 * 4 * 1_152 * 4
    select_hooks = 1_872 * 4 * 4 * 1_152 * 4
    fit_live_u = 3_744 * 4 * 2 * 1_152 * 4
    select_live_u = 1_872 * 4 * 2 * 1_152 * 4
    assert (fit_hooks, select_hooks) == (276_037_632, 138_018_816)
    assert (fit_live_u, select_live_u) == (138_018_816, 69_009_408)
    assert fit_logits + select_logits + fit_hooks + select_hooks + fit_live_u + select_live_u == 5_141_200_896

    old_fit = 3_744 * 4 * 50_257 * 4
    old_select = 1_872 * 4 * 50_257 * 4
    assert fit_logits - old_fit == 2_815_488
    assert select_logits - old_select == 1_407_744
    assert (fit_logits + select_logits) - (old_fit + old_select) == 4_223_232


def test_supersession_covers_all_full_logit_consumers_without_slicing() -> None:
    text = blob(AMENDMENT).decode()
    required = (
        "[b,50304]", "[N_d,4,50304]", "native_minus_replay",
        "score_minus_replay", "payload_minus_replay", "joint_minus_replay",
        "all 50,304 checkpoint logits", "No slicing", "Every saved `vocab_size` field",
        "FIT-frozen vocabulary scales", "nonfinite-mask rule", "[b,50304]",
        "little-endian float32", "float64 aggregate arithmetic",
    )
    assert all(term in text for term in required)
    assert "j=0}^{50303}" in text
    assert "50,257 coordinates as “full vocabulary”" in text
    assert "is not authorized here" in text


def test_narrow_boundary_preserves_science_and_price() -> None:
    text = blob(AMENDMENT).decode()
    frozen = (
        "no row, direction, endpoint, semantic role, site, machine arm, centered-factor formula",
        "support rule, factor-transport check, operational tolerance, target/control cell, bootstrap identity or draw, scientific",
        "threshold, FIT-first opening rule, terminal precedence, publication rule, claim boundary, or call price",
        "639 FIT, 322 SELECT, 961 maximum, zero backward passes, and zero weight updates",
        "FINAL and OOD remain",
    )
    assert all(term in text for term in frozen)
    assert "The six findings in the implementation review remain blocking" in text
    assert "None of those implementation repairs is made or" in text
    assert "authorized by this document" in text


def test_lineage_hashes_are_exact() -> None:
    text = blob(AMENDMENT).decode()
    for digest in (
        "9b8e4ce54d1b34d650ef088f841672cf01a4482257446b611ba37e1353a457cf",
        "3f8a559a14015498d375ba75271cf57647b9cc9841ef32b1e9e32406abf71323",
        FACADE_SHA256,
    ):
        assert digest in text
