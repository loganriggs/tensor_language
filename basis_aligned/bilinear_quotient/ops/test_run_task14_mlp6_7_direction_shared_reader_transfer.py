import run_task14_mlp6_7_direction_shared_reader_transfer as run

def test_price_is_one_role_and_one_gradient_forward():
 assert run.derive_price()=={"physical_model_forwards":2,"example_evaluations":128,"backwards":1,"causal_interventions":0,"parameter_updates":0}

def test_plan_is_leave_one_out_and_not_rank_fit():
 p=run.compile_plan(); assert "leave-one-row-out" in p["reader_rule"] and "global" in p["control"]

def test_parent_has_complete_actual_lattice():
 assert len(run._actual())==512
