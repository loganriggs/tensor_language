import importlib.util
from pathlib import Path


SOURCE = Path(__file__).with_name("causal_action_coverage_audit_rung497.py")
SPEC = importlib.util.spec_from_file_location("r497", SOURCE)
R497 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(R497)


def test_manifest_is_unique_and_lawful():
    ids = [item["evidence_id"] for item in R497.EVIDENCE]
    assert len(ids) == len(set(ids))
    assert all(set(item["actions"]) <= R497.ALLOWED_ACTIONS for item in R497.EVIDENCE)
    assert {item["family"] for item in R497.EVIDENCE} == {
        "equality_attention_terms",
        "equality_correction_sites",
        "mlp0_branches_block1",
        "mlp1_write_adjustments",
        "attention1_factor_writes",
    }


def test_synthetic_action_complete_family_passes():
    rows = []
    for action in R497.FINITE_ACTIONS:
        rows.append(R497.record(
            f"synthetic_{action}", "synthetic", "unused.json", (action,), "ok",
            candidates=2, sites="x", per_example=True, document_splits=2,
            same_action_semantics=True, physical_suffix=True, dedicated_task=True,
            circuit_scope="62_heldout_circuits", heldout_circuits=True,
            sequential_actions=(action == "compose")))
    score = R497.score_families(rows)["synthetic"]
    assert score["archive_ready"] is True
    assert score["transition_refinement_ready"] is True
    assert not score["missing_qualified_actions"]


def test_current_manifest_requires_prospective_collection():
    records = R497.validate_evidence()
    score = R497.score_families(records)
    assert not any(item["archive_ready"] for item in score.values())
    assert not any(item["transition_refinement_ready"] for item in score.values())
    equality = score["equality_attention_terms"]
    assert equality["has_known_positive"] is True
    assert equality["has_known_negative"] is True
    assert equality["has_heldout_circuit_record"] is False
