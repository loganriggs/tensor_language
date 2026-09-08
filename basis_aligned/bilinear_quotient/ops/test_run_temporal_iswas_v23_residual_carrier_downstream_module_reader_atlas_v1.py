import torch
import run_temporal_iswas_v23_residual_carrier_downstream_module_reader_atlas_v1 as experiment

def test_metrics_exact_vector():
    value=torch.tensor([1.,-2.,3.]); selected=torch.ones(3,dtype=torch.bool)
    report=experiment.metrics(torch,value,value,selected)
    assert report["relative_residual"] == 0 and report["direction_fraction"] == 1

def test_price_and_module_inventory():
    assert experiment.PRICE["model_forwards_exact"] == 4 + 2*len(experiment.MODULES) == 28
    assert experiment.PRICE["sequence_evaluations_exact"] == 28*64
    assert experiment.MODULES[0] == "A12" and experiment.MODULES[-1] == "M17"

def test_executor_is_fail_closed_before_factorial_binding():
    assert experiment.EXPECTED_FACTORIAL_RESULT_SHA256 is None
