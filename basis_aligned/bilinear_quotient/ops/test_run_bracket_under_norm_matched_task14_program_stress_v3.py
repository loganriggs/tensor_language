import pytest
import run_bracket_under_norm_matched_task14_program_stress_v3 as run

def test_gain_is_exactly_artifact_derived_and_outcome_free():
 assert run.derive_gain()==pytest.approx(run.FROZEN_GAIN,abs=1e-12)
 plan=run.compile_plan();assert plan["gain_source"]=="artifact median L2 norm ratio; zero outcome values"
 assert plan["price"]["physical_model_forwards"]==1 and plan["price"]["example_evaluations"]==576

def test_scope_does_not_reopen_original_gain_composition():
 assert "original-gain reverse" in run.compile_plan()["scope"]
