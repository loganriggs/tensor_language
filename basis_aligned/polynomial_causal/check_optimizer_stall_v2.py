"""A nonstationary optimizer that makes no updates must never be called converged."""
import contextlib,io,json
from pathlib import Path
from unittest.mock import patch
import torch
from convergent_quadratic_fit_v2 import advance

class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.x=torch.nn.Parameter(torch.ones(1,dtype=torch.float64))

class Objective:
    def loss(self,m):return .5*m.x.square().sum(),m.x.detach()[:,None],torch.eye(1,dtype=torch.float64)
    def diagnostics(self,m):
        m.zero_grad(set_to_none=True);loss,w,_=self.loss(m);loss.backward()
        return dict(squared_relative_error=float(loss.detach()),optimization_loss=float(loss.detach()),
            captured_energy_fraction=1-float(loss.detach()),relative_stationarity=float(m.x.grad.norm())*2,
            gradient_max_abs=float(m.x.grad.abs().max()),gram_condition=1.),w

class NoStep(torch.optim.SGD):
    def step(self,closure):return closure()

def main():
    with patch('torch.optim.LBFGS',lambda params,**kwargs:NoStep(params,lr=0.)),contextlib.redirect_stdout(io.StringIO()):
        result=advance(Model(),Objective(),checkpoint={'phase':'lbfgs'},seconds=10)
    assert not result['converged'] and result['terminal_reason']=='line_search_stalled'
    assert result['chunk_updates']==25 and result['lbfgs_done']==25 and result['history'][-1]['gradient_max_abs']==1.
    receipt=dict(passed=True,reason=result['terminal_reason'],iterations=result['lbfgs_done'],
        converged=result['converged'],gradient_max_abs=result['history'][-1]['gradient_max_abs'])
    with Path(__file__).with_name('OPTIMIZER_STALL_V2_CONTROL.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
