import torch
from types import SimpleNamespace
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

def test_block_correction_preserves_auxiliary_stream_and_cleans_hook():
    class Block:
        def __init__(self): self.hooks=[]
        def register_forward_hook(self, hook):
            self.hooks.append(hook); return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
        def run(self, output):
            for hook in tuple(self.hooks):
                changed=hook(self,(),output)
                if changed is not None: output=changed
            return output
    block=Block(); layers=[SimpleNamespace() for _ in range(18)]; layers[11]=block
    model=SimpleNamespace(transformer=SimpleNamespace(h=layers))
    x,v1=torch.ones(2,3,4),torch.tensor([7.])
    result,calls=experiment.add_block_correction(model,lambda:block.run((x,v1)),torch.full_like(x,2))
    assert torch.equal(result[0],torch.full_like(x,3)) and result[1] is v1
    assert calls == {"block11":1} and block.hooks == []
