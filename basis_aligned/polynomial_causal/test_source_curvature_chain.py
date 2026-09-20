"""Planted independent chain-rule check for local Hessian source attribution."""
import json
from pathlib import Path
import torch

def main():
    torch.manual_seed(20260920);torch.set_num_threads(2);d=4;p=3;dtype=torch.float64
    h=torch.randn(d,dtype=dtype);K=torch.randn(d,p,dtype=dtype);w=torch.randn(d,dtype=dtype);M=torch.randn(d,d,dtype=dtype)
    def first(x):return x+(M@x).square()
    def second(x):return x+x.square()/(1+x.square().mean())
    def read(x):return torch.tanh(w@x)
    def whole(a):return read(second(first(h+K@a)))
    a0=torch.zeros(p,dtype=dtype);total=torch.autograd.functional.hessian(whole,a0)
    x1=first(h);x2=second(x1);J1=torch.autograd.functional.jacobian(first,h);J2=torch.autograd.functional.jacobian(second,x1)
    q2=torch.autograd.functional.jacobian(read,x2);q1=J2.T@q2
    Hfirst=torch.autograd.functional.hessian(lambda x:q1@first(x),h)
    Hsecond=torch.autograd.functional.hessian(lambda x:q2@second(x),x1)
    Hread=torch.autograd.functional.hessian(read,x2)
    terms=[K.T@Hfirst@K,(J1@K).T@Hsecond@(J1@K),(J2@J1@K).T@Hread@(J2@J1@K)]
    error=float((sum(terms)-total).abs().max());assert error<1e-10
    omitted=float((sum(terms[:2])-total).norm());assert omitted>1e-6
    result=dict(chain_rule_max_abs=error,omitted_readout_error=omitted,terms=[x.tolist() for x in terms],scope='CPU planted nonlinear chain. Exact local readers and propagated source Jacobians are necessary; native full-suffix decomposition pending.')
    Path(__file__).with_name('SOURCE_CURVATURE_CHAIN_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result['chain_rule_max_abs'],result['omitted_readout_error'])
if __name__=='__main__':main()
