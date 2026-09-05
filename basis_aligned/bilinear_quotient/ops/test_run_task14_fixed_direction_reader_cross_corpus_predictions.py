import run_task14_fixed_direction_reader_cross_corpus_predictions as run
def test_price_has_no_target_tail_or_causal_work():
 assert run.derive_price()=={"physical_model_forwards":1,"example_evaluations":96,"backwards":0,
  "causal_interventions":0,"sealed_predictions":512,"parameter_updates":0}
def test_plan_binds_fixed_artifact_and_closes_outcomes():
 p=run.compile_plan(); assert p["reader_artifact_sha256"]==run.READER_SHA256 and p["target_tail_backwards"]==0 and p["causal_outcomes_opened"] is False
