import run_task14_fixed_reader_guided_margin_edit as run
def test_support_and_price_are_frozen():
 assert len(run._prediction_rows())==278
 assert run.derive_price()=={"physical_model_forwards":5,"example_evaluations":930,"causal_installations":834,"maximum_patch_chunk_rows":256,"patch_chunks":4,"backwards":0,"parameter_updates":0}
def test_plan_gain_range_is_safe():
 p=run.compile_plan(); assert 0<p["gain_range"][0]<=p["gain_range"][1]<=.8
def test_exact_guided_target_passes_and_beats_variable_half():
 evidence=[]
 for d in ("plural_to_singular","singular_to_plural"):
  for t in ("above_inside","inside_above"):
   for i in range(70):
    desired=-.04 if d=="plural_to_singular" else .04
    evidence.append({"row_id":f"{d}-{t}-{i}","direction":d,"template":t,"background":str(i),"desired_q":desired,"alpha":.4,"guided_q":desired,"half_q":desired*1.5})
 scored=run.score(evidence[:278],{"x":0.0}); assert all(scored["predictions"].values())
