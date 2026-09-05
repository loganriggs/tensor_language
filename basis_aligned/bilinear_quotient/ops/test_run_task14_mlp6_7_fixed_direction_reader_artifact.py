import run_task14_mlp6_7_fixed_direction_reader_artifact as run

def test_price_exports_only_two_vectors():
 assert run.derive_price()=={"physical_model_forwards":2,"example_evaluations":128,"backwards":1,
  "causal_interventions":0,"parameter_updates":0,"stored_scalars":256}

def test_plan_is_target_free_direction_reader_export():
 p=run.compile_plan(); assert p["reader_width"]==128 and len(p["directions"])==2
 assert "target-free" in p["source"]
