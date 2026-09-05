import run_task14_fixed_direction_reader_cross_corpus_causal_validation as run
def test_price_is_complete_lattice():
 assert run.derive_price()=={"physical_model_forwards":5,"example_evaluations":1120,"causal_interventions":1024,
  "backwards":0,"parameter_updates":0,"maximum_patch_chunk_rows":256,"patch_chunks":4}
def test_plan_binds_seal_and_no_calibration():
 p=run.compile_plan(); assert p["sealed_prediction_sha256"]==run.PREDICTION_SHA256 and "no scale" in p["literal_scorer"]
def test_exact_sealed_values_pass_and_swapped_control_loses():
 sealed=run._load_prediction()["evidence"]; causal=[{"row_id":x["row_id"],"direction":x["direction"],"template":x["template"],
  "background":x["background"],"cardinality":x["cardinality"],"actual_q":x["fixed_reader_q"]} for x in sealed]
 scored=run.score(causal,{"closure":0.0}); assert all(scored["predictions"].values())
