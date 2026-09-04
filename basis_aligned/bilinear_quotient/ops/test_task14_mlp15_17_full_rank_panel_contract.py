from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = ROOT / "ops/task14_mlp15_17_full_rank_panel_contract_v1.json"


def _load(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _target_class(row: dict) -> str:
    if row["expected_relation"] == "same_subject_zero_projected_effect":
        return "control"
    if row["arm"] == "answer_change" and row["matching"] == "paired":
        return "paired"
    if row["arm"] in {"answer_change", "P_positive_transfer"} \
            and row["matching"].startswith("cross_noun"):
        return "cross_noun"
    if row["arm"] == "cross_syntax":
        return "literal_cross_syntax"
    if row["arm"] in {"C_to_ordinary_singular", "ordinary_singular_to_C"}:
        return "complete_subject_transfer"
    raise AssertionError(f"unclassified target relation: {row}")


def test_all_phase0_sources_match_including_audit_and_cached_factorial():
    contract = _load(CONTRACT_PATH)
    for source in contract["sources"].values():
        assert _sha256(ROOT / source["path"]) == source["sha256"]


def test_relation_ordinals_roles_and_semantic_classes_recompile_exactly():
    contract = _load(CONTRACT_PATH)
    donors = _load(ROOT / contract["sources"]["donors"]["path"])
    endpoints = {row["endpoint_id"]: row for row in donors["endpoints"]}
    audit = _load(ROOT / contract["sources"]["relation_audit"]["path"])

    for split in ("FIT", "SELECT"):
        groups = set(audit["inner_splits"][split]["group_numbers"])
        rows = []
        for row in donors["records"]:
            if row["partition"] != "DISCOVERY":
                continue
            target_group = endpoints[row["target_endpoint_id"]]["group_number"]
            donor_group = endpoints[row["donor_endpoint_id"]]["group_number"]
            if target_group in groups and donor_group in groups:
                rows.append(row)
        rows.sort(key=lambda row: row["ordinal"])
        frozen = audit["inner_splits"][split]
        assert len(rows) == frozen["relation_count"]
        assert _canonical_sha([row["ordinal"] for row in rows]) == frozen["ordinal_sha256"]
        roles = Counter(
            "target" if row["expected_relation"] == "opposite_subject_toward_donor"
            else "control" if row["expected_relation"] == "same_subject_zero_projected_effect"
            else "invalid"
            for row in rows
        )
        assert roles == Counter(target=frozen["target_count"], control=frozen["control_count"])
        assert Counter(_target_class(row) for row in rows if _target_class(row) != "control") \
            == Counter(frozen["target_class_counts"])
        assert Counter(row["arm"] for row in rows if _target_class(row) == "control") \
            == Counter(frozen["control_arm_counts"])


def test_full_rank_removal_and_sufficiency_have_frozen_opposite_signs():
    rng = np.random.default_rng(1401517)
    z_b = rng.standard_normal(13)
    z_h = rng.standard_normal(13)
    delta = z_h - z_b

    # A nonlinear fake downstream function makes the endpoint/sign test stronger
    # than an accidental linearity check.
    def fake_f(z):
        return np.array([np.sin(z).sum(), np.square(z).sum(), np.max(z)])

    e_full = fake_f(z_b) - fake_f(z_h)
    e_remove = fake_f(z_h - delta) - fake_f(z_h)
    e_suff = fake_f(z_b + delta) - fake_f(z_b)
    np.testing.assert_allclose(z_h - delta, z_b, atol=1e-12, rtol=0)
    np.testing.assert_allclose(z_b + delta, z_h, atol=1e-12, rtol=0)
    np.testing.assert_allclose(e_remove, e_full, atol=1e-12, rtol=0)
    np.testing.assert_allclose(e_suff, -e_full, atol=1e-12, rtol=0)


def test_exact_three_term_bilinear_expansion_and_eight_subset_lattice():
    rng = np.random.default_rng(151700)
    wl = rng.standard_normal((17, 7))
    wr = rng.standard_normal((17, 7))
    x_b = rng.standard_normal(7)
    dx = rng.standard_normal(7)
    z_b = (wl @ x_b) * (wr @ x_b)
    z_h = (wl @ (x_b + dx)) * (wr @ (x_b + dx))
    terms = (
        (wl @ x_b) * (wr @ dx),
        (wl @ dx) * (wr @ x_b),
        (wl @ dx) * (wr @ dx),
    )
    np.testing.assert_allclose(z_h - z_b, sum(terms), atol=1e-12, rtol=1e-12)
    lattice = {
        mask: z_b + sum((terms[j] for j in range(3) if mask & (1 << j)), start=np.zeros_like(z_b))
        for mask in range(8)
    }
    np.testing.assert_allclose(lattice[0], z_b, atol=0, rtol=0)
    np.testing.assert_allclose(lattice[7], z_h, atol=1e-12, rtol=1e-12)


def test_fake_backend_joint_recomputes_layer17_after_layer15_reset():
    # A small depth-ordered fake model. Reusing cached z17_H after the z15 reset
    # would fail the explicit inequality and endpoint checks below.
    z15_b = np.array([0.2, -0.4])
    z15_h = np.array([0.8, 0.1])

    def z17_after(z15):
        x = np.array([[1.0, 0.3], [-0.2, 0.7]]) @ z15 + np.array([0.1, -0.3])
        return x * x

    z17_b = z17_after(z15_b)
    z17_h = z17_after(z15_h)
    z17_after_15_reset = z17_after(z15_h - (z15_h - z15_b))
    assert not np.array_equal(z17_after_15_reset, z17_h)
    np.testing.assert_allclose(z17_after_15_reset, z17_b, atol=1e-12, rtol=0)
    delta17_live = z17_after_15_reset - z17_b
    np.testing.assert_allclose(z17_after_15_reset - delta17_live, z17_b, atol=0, rtol=0)


def test_joint_recompute_provenance_and_direction_gate_are_unambiguous():
    contract = _load(CONTRACT_PATH)
    provenance = contract["depth_ordered_joint"]["required_hook_provenance"]
    assert provenance["event_count_each_per_relation"] == 1
    assert provenance["ordered_events_per_relation"] == [
        "mlp15_product_enter",
        "mlp15_reset_applied",
        "mlp17_product_enter_after_mlp15",
        "mlp17_live_product_captured",
        "mlp17_reset_applied",
    ]
    assert "cached z17_H" in provenance["capture_rule"]
    direction = contract["opposing_predictions"]["direction_specific_response"]
    assert "depth-ordered joint" in direction["gated_effect"]
    assert "do not determine" in direction["gated_effect"]


def test_phase0_is_no_optimizer_and_price_is_exactly_bounded():
    contract = _load(CONTRACT_PATH)
    assert contract["execution_authorized"] is False
    assert contract["screen_only"] is True
    assert contract["optimizer"] is None
    assert contract["candidate_ranks"] is None
    price = contract["price"]
    assert price["model_forward_calls_max"] == 4 + 7 * 10 == 74
    assert price["sequence_examples_max"] == 128 + 7 * 298 == 2214
    assert price["backward_calls"] == price["optimizer_updates"] == 0
    phase0b = contract["exact_bilinear_expansion"]["phase0B_price"]
    assert phase0b["additional_relation_forward_calls_max_at_batch32"] == 12 * 10 == 120
    assert phase0b["additional_sequence_examples_max"] == 12 * 298 == 3576


def test_claims_are_quotient_aware_and_unrelated_control_is_deferred():
    contract = _load(CONTRACT_PATH)
    quotient = contract["product_space_identifiability"]
    assert quotient["minimum_nullity"] == 4608 - 1152 == 3456
    assert "unique hidden product basis" in quotient["forbidden_claim"]
    unrelated = contract["unrelated_behavior_control"]
    assert unrelated["phase0_status"] == "deferred"
    assert "liveness floor" in unrelated["future_entry_condition"]
