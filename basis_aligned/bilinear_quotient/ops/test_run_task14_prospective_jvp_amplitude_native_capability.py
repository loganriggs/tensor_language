import circuit_fast_screen_candidate_task14_prospective_jvp_amplitude as authority
import run_task14_prospective_jvp_amplitude_native_capability as run


def test_authority_is_frozen_balanced_and_new():
    rows = authority.build_rows()
    assert len(rows) == 32
    assert authority.canonical_sha256(rows) == authority.EXPECTED_AUTHORITY_SHA256
    assert {row["template_id"] for row in rows} == {"above_below", "below_above"}


def test_capability_has_twelve_balanced_cells():
    gate = run.build_gate()
    assert len(gate.cells) == 12
    assert {cell.expected_count for cell in gate.cells} == {8}


def test_plan_is_native_only_and_causal_candidate_scoped():
    plan = run.compile_plan()
    assert plan["native_only"] is True
    assert plan["causal_candidate_id"] == authority.CAUSAL_CANDIDATE_ID
    assert plan["endpoint_evaluations"] == 96
