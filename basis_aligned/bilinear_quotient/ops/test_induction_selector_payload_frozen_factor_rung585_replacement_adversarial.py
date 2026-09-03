"""Independent specification tests for the frozen R585 replacement package.

These tests intentionally exercise only the committed model-free authority and
manifest.  They neither import nor inspect a future R585 model runner.
"""

# BQLANE: cpu

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
POLY = ROOT.parent / "polynomial_causal"
MANIFEST_PATH = OPS / "induction_selector_payload_frozen_factor_rung585_manifest.py"
ROWS_PATH = ROOT / "induction_selector_payload_three_source_rows_rung578.json"
DRYRUN_PATH = ROOT / "induction_selector_payload_frozen_factor_rung585_manifest_dryrun.json"
LOCK_PATH = ROOT / "induction_selector_payload_frozen_factor_rung585_dependency_lock.json"
AMENDMENT_PATH = POLY / "INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_REPLACEMENT_AMENDMENT.md"

FROZEN_SHA256 = {
    MANIFEST_PATH: "7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962",
    ROWS_PATH: "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6",
    DRYRUN_PATH: "dc81109bed0ef44c51224988a53d57143751a3f078a889c156a7a8862e52114f",
    LOCK_PATH: "908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7",
    AMENDMENT_PATH: "98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf",
}
UPSTREAM_SHA256 = {
    "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_results.json":
        "14e7414bc7cf6b4a6a221079ac378752602b021b8b411124149dcc2c311666b8",
    "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_rung586_receipt.json":
        "afd7533b1838b7d230858696a059f9c3a5903e75f031aa0c86f175f4bc0d9384",
    "basis_aligned/bilinear_quotient/induction_selector_payload_native_capability_audit_rung587.json":
        "72f0261fe32aa3d048c442ea1c08af932af6a368894610833e79aaaabf98bfe9",
}
SITES = ("L5H5", "L7H3", "L8H3", "L8H4")
ROLES = ("A", "C")
SCALE_KINDS = ("insert_norm", "margin_logit", "vocabulary_logit_rms")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()).hexdigest()


def _load_manifest():
    name = "r585_replacement_manifest_adversarial_target"
    spec = importlib.util.spec_from_file_location(name, MANIFEST_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    return _load_manifest()


@pytest.fixture(scope="module")
def document():
    return json.loads(ROWS_PATH.read_text())


@pytest.fixture(scope="module")
def included(document, manifest):
    return [row for row in document["rows"]
            if row["split"] in manifest.SPLITS
            and row["family_id"] in manifest.INCLUDED_FAMILIES]


def _validate_semantic_endpoint(row: dict, prefix: str) -> None:
    ids = row[f"{prefix}_ids"]
    structure = row[f"{prefix}_structure"]
    source_positions = structure["source_positions"]
    payload_positions = structure["payload_positions"]
    if len(source_positions) != 2 or len(payload_positions) != 2:
        raise ValueError("the A/C semantic-role arrays must each have length two")
    if len(set(source_positions)) != 2 or len(set(payload_positions)) != 2:
        raise ValueError("semantic A/C positions are ambiguous")
    if list(structure["pair_order"]) not in [
        ["A", "C", "N"], ["C", "N", "A"], ["N", "A", "C"]
    ]:
        raise ValueError("pair-order metadata drifted")
    if [ids[index] for index in source_positions] != structure["source_ids"]:
        raise ValueError("source tokens disagree with semantic positions")
    if [ids[index] for index in payload_positions] != structure["payload_ids"]:
        raise ValueError("payload tokens disagree with semantic positions")
    if any(payload != source + 1
           for source, payload in zip(source_positions, payload_positions)):
        raise ValueError("payload is not the registered successor of its source")
    if ids[structure["query_position"]] != structure["query_id"]:
        raise ValueError("query token disagrees with its semantic position")
    if len(set(structure["source_ids"])) != 2:
        raise ValueError("canonical A/C source tokens are ambiguous")
    equality_count = sum(
        source == structure["query_id"] for source in structure["source_ids"]
    )
    if equality_count not in (0, 1):
        raise ValueError("endpoint has ambiguous canonical equality support")


def _condition(row: dict, prefix: str) -> str:
    return f"s{row[prefix + '_selector']}p{row[prefix + '_payload_assignment']}"


def _independent_cell_records(rows: list[dict], manifest) -> list[dict]:
    buckets = {}
    for row in rows:
        for direction, recipient in (("base_to_donor", "base"),
                                     ("donor_to_base", "donor")):
            key = (
                row["split"], row["family_id"], row["family_variant"],
                _condition(row, recipient), direction,
            )
            buckets.setdefault(key, []).append((row["group_id"], row["row_id"]))
    output = []
    for key, members in sorted(buckets.items()):
        groups = sorted(group for group, _ in members)
        assert len(groups) == len(set(groups))
        output.append({
            "cell_id": "|".join(key),
            "split": key[0], "family": key[1], "variant": key[2],
            "recipient_condition": key[3], "direction": key[4],
            "role": "target" if key[1] in manifest.TARGET_FAMILIES else "control",
            "group_ids": groups,
        })
    return output


def _bootstrap_specs(family: str, direction: str) -> list[tuple[str, str]]:
    common = ("denominator_mean", "numerator_mean", "donor_ce_mean")
    if family == "two_valid_sources_selector_swap":
        return [(arm, metric) for arm in ("score", "joint") for metric in common]
    if family == "payload_swap_match_preserved":
        return [("score", metric) for metric in common[:2]] + [
            (arm, metric) for arm in ("payload", "joint") for metric in common
        ]
    if family == "match_break_payload_preserved":
        result = [(arm, metric) for arm in ("score", "joint") for metric in common]
        if direction == "base_to_donor":
            result += [("payload", metric) for metric in common[:2]]
        return result
    if family == "selector_payload_joint_answer_preserved":
        return [
            ("score", "single_score_harm_mean"),
            ("payload", "single_payload_harm_mean"),
            ("joint", "factorial_interaction_mean"),
        ]
    raise ValueError(f"not a target family: {family}")


def _denominator_status(denominators: list[float], lower95: float) -> str:
    values = np.asarray(denominators, dtype=np.float64)
    if not np.isfinite(values).all() or not math.isfinite(lower95):
        return "native_denominator_null"
    if float(values.mean()) <= 0 or lower95 <= 0 or float(np.median(values)) <= 0:
        return "native_denominator_null"
    return "valid"


def _validate_expanded_scale_lookup(records: list[dict]) -> None:
    keys = set()
    for record in records:
        key = (record["control_cell_id"], record["arm"], record["scale_kind"])
        if key in keys:
            raise ValueError("duplicate control scale lookup")
        keys.add(key)
        if record["scale_kind"] != record["fit_target_scale_kind"]:
            raise ValueError("control and target scale units collide")
        if not record["fit_target_cell_id"].startswith("FIT|"):
            raise ValueError("a control scale is not FIT-frozen")


def test_exact_committed_package_hashes_and_cpu_boundary(manifest):
    assert {path: _sha256(path) for path in FROZEN_SHA256} == FROZEN_SHA256
    source = MANIFEST_PATH.read_text()
    assert "import torch" not in source
    assert "load_bilin18" not in source
    assert manifest.BOOTSTRAPS == 2_000


def test_semantic_roles_are_unambiguous_in_exact_r578(included):
    for row in included:
        _validate_semantic_endpoint(row, "base")
        _validate_semantic_endpoint(row, "donor")
    assert len(included) == 2_808


def test_planted_ambiguous_canonical_tokens_are_rejected(included):
    planted = copy.deepcopy(included[0])
    structure = planted["base_structure"]
    structure["source_ids"][1] = structure["source_ids"][0]
    planted["base_ids"][structure["source_positions"][1]] = structure["source_ids"][0]
    with pytest.raises(ValueError, match="ambiguous"):
        _validate_semantic_endpoint(planted, "base")


def test_metadata_drift_changes_the_frozen_direction_manifest(included, manifest):
    exact = manifest.build_authority_manifest(included)
    planted = copy.deepcopy(included)
    planted[0]["group_id"] = "planted-drift-group"
    drifted = manifest.build_authority_manifest(planted)
    assert _canonical_hash(exact["directions"]) != _canonical_hash(drifted["directions"])
    dryrun = json.loads(DRYRUN_PATH.read_text())
    assert _canonical_hash(exact["directions"]) == dryrun["direction_manifest_sha256"]


def test_recipient_donor_conditions_and_match_signs(included):
    expected = {
        ("two_valid_sources_selector_swap", "payload_assignment_0"): ("s0p0", "s1p0"),
        ("two_valid_sources_selector_swap", "payload_assignment_1"): ("s0p1", "s1p1"),
        ("payload_swap_match_preserved", "selector_0"): ("s0p0", "s0p1"),
        ("payload_swap_match_preserved", "selector_1"): ("s1p0", "s1p1"),
        ("selector_payload_joint_answer_preserved", "payload_B"): ("s0p0", "s1p1"),
        ("selector_payload_joint_answer_preserved", "payload_D"): ("s1p0", "s0p1"),
    }
    for key, pair in expected.items():
        rows = [row for row in included
                if (row["family_id"], row["family_variant"]) == key]
        assert {(_condition(row, "base"), _condition(row, "donor"))
                for row in rows} == {pair}
    match = [row for row in included
             if row["family_id"] == "match_break_payload_preserved"]
    assert all(sum(source == row["base_structure"]["query_id"]
                   for source in row["base_structure"]["source_ids"]) == 1
               for row in match)
    assert all(sum(source == row["donor_structure"]["query_id"]
                   for source in row["donor_structure"]["source_ids"]) == 0
               for row in match)
    assert all(row["base_answer_id"] == row["donor_answer_id"] for row in match)


def test_planted_denominator_zero_has_frozen_terminal_class():
    assert _denominator_status([0.0] * 72, 0.0) == "native_denominator_null"
    assert _denominator_status([1.0, -1.0] * 36, 0.1) == "native_denominator_null"
    assert _denominator_status([1.0] * 72, 0.5) == "valid"


def test_independent_cell_counts_and_structural_identity_counts(included, manifest):
    cells = _independent_cell_records(included, manifest)
    for split in manifest.SPLITS:
        targets = [cell for cell in cells if cell["split"] == split and cell["role"] == "target"]
        controls = [cell for cell in cells if cell["split"] == split and cell["role"] == "control"]
        assert (len(targets), len(controls)) == (20, 32)
        assert {len(cell["group_ids"]) for cell in targets + controls} == \
            ({72} if split == "FIT" else {36})
    built = manifest.build_cell_manifests(manifest.build_authority_manifest(included))
    assert {split: sum(identity["cell_id"].startswith(split + "|")
                       for identity in built["structural_identities"])
            for split in manifest.SPLITS} == {"FIT": 32, "SELECT": 32}
    assert {split: sum(cell["split"] == split
                       for cell in built["eligible_control_arm_cells"])
            for split in manifest.SPLITS} == {"FIT": 88, "SELECT": 88}


def test_role_and_site_operation_census_detects_omissions(manifest):
    authority = manifest.build_authority_manifest()
    operations = {
        (endpoint["split"], endpoint["endpoint_id"], site, role)
        for endpoint in authority["endpoints"] for site in SITES for role in ROLES
    }
    assert len(operations) == (1_728 + 864) * 4 * 2
    without_one_role = {item for item in operations if item[3] != "C"}
    without_one_site = {item for item in operations if item[2] != "L8H4"}
    assert len(without_one_role) != len(operations)
    assert len(without_one_site) != len(operations)
    assert {site for _, _, site, _ in operations} == set(SITES)
    assert {role for _, _, _, role in operations} == set(ROLES)


def test_three_scale_units_expand_without_control_target_collisions(manifest):
    lookup = manifest.build_control_scale_lookup()
    expanded = [
        {**record, "scale_kind": kind, "fit_target_scale_kind": kind}
        for record in lookup for kind in SCALE_KINDS
    ]
    assert len(lookup) == 192
    assert len(expanded) == 576
    _validate_expanded_scale_lookup(expanded)
    collision = copy.deepcopy(expanded)
    collision[0]["fit_target_scale_kind"] = "margin_logit"
    collision[0]["scale_kind"] = "insert_norm"
    with pytest.raises(ValueError, match="units collide"):
        _validate_expanded_scale_lookup(collision)


def test_bootstrap_identity_is_independently_reconstructed(included, manifest):
    cells = _independent_cell_records(included, manifest)
    ids = []
    for cell in cells:
        if cell["role"] != "target":
            continue
        for arm, metric in _bootstrap_specs(cell["family"], cell["direction"]):
            ids.append(f"{cell['cell_id']}|{arm}|{metric}")
    ids.sort()
    dryrun = json.loads(DRYRUN_PATH.read_text())
    assert len(ids) == 248 and len(set(ids)) == 248
    assert sum(value.startswith("FIT|") for value in ids) == 124
    assert _canonical_hash(ids) == dryrun["bootstrap_cell_ids_sha256"]
    sentinel = dryrun["bootstrap_sentinel"]
    groups = next(cell["group_ids"] for cell in cells
                  if sentinel["cell_id"].startswith(cell["cell_id"] + "|"))
    raw = bytearray()
    for replicate in range(2_000):
        for draw in range(len(groups)):
            payload = (
                f"a8-r585-replacement-group-bootstrap-v1:"
                f"{sentinel['cell_id']}:{replicate}:{draw}"
            ).encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % len(groups)
            raw.extend(index.to_bytes(2, "big"))
    assert hashlib.sha256(raw).hexdigest() == sentinel["big_endian_uint16_draw_sha256"]


def test_dependency_lock_exact_bytes_verdicts_and_tampering(manifest):
    lock = json.loads(LOCK_PATH.read_text())
    assert lock["r586_result_sha256"] == UPSTREAM_SHA256[lock["r586_result_path"]]
    assert lock["r586_receipt_sha256"] == UPSTREAM_SHA256[lock["r586_receipt_path"]]
    assert lock["r587_audit_sha256"] == UPSTREAM_SHA256[lock["r587_audit_path"]]
    assert manifest.validate_dependency_lock(lock, UPSTREAM_SHA256)["runnable"] is True
    tampered_hashes = dict(UPSTREAM_SHA256)
    tampered_hashes[lock["r586_result_path"]] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        manifest.validate_dependency_lock(lock, tampered_hashes)
    changed = copy.deepcopy(lock)
    changed["r586_verdict"] = "scientific_null"
    result = manifest.validate_dependency_lock(changed, UPSTREAM_SHA256)
    assert result["terminal"] == "not_executed_upstream_dependency"


def test_execution_accounting_rederived_from_census(manifest):
    authority = manifest.build_authority_manifest()
    accounting = manifest.phase_accounting(authority)
    assert math.ceil(1_728 / 32) + 3 * math.ceil(3_744 / 32) \
        + math.ceil(1_728 / 32) == 459
    assert math.ceil(864 / 32) + 3 * math.ceil(1_872 / 32) \
        + math.ceil(864 / 32) == 231
    assert accounting["phases"]["FIT"]["phase_max_forwards"] == 459
    assert accounting["phases"]["SELECT"]["phase_max_forwards"] == 231
    assert accounting["total_max_forwards"] == 690
    assert accounting["model_backwards"] == accounting["weight_updates"] == 0


def test_fit_first_terminal_precedence_and_licensed_conclusion_are_exact():
    precedence = [
        "not_executed_upstream_dependency", "integrity_abort", "invalid_instrument",
        "native_denominator_or_scale_null", "factor_capacity_null",
        "factorization_not_identified", "insufficient_active_controls",
        "broad_contextual_equality_write",
        "held_operational_selector_payload_factorization",
    ]
    failures = {"factor_capacity_null", "invalid_instrument", "broad_contextual_equality_write"}
    winner = next(label for label in precedence if label in failures)
    assert winner == "invalid_instrument"
    text = AMENDMENT_PATH.read_text()
    assert "SELECT opens exactly once only if every FIT scientific and instrument clause passes" in text
    assert "held_operational_selector_payload_factorization" in text
    for forbidden_claim in (
        "unique Q/K features", "OOD generalization", "weight-level compiler",
        "individual-site necessity", "selective removal",
    ):
        assert forbidden_claim in text

