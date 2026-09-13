"""Independent finite-difference check of the code-eliminated bank derivative."""
import json
from pathlib import Path
import torch
from check_shared_local_subspaces_v1 import fixture
from shared_local_subspaces_v1 import initialize,encode,objective,parts,fit,stationarity


def main():
    torch.set_num_threads(2)
    x,g,k,r=fixture()
    state=initialize(x,g,k,r,0)
    p=state['global_bank']
    torch.manual_seed(9421)
    h=torch.randn(x.shape[1],x.shape[1],dtype=x.dtype)
    h=(h-h.T)/2;h=h/h.norm()
    residual=sum(parts(state))-x
    grad=2*state['global_codes'].T@residual/x.square().sum()
    expected=float((grad*(p@h)).sum())
    eps=1e-5
    plus=encode(x,p@torch.matrix_exp(eps*h),state['local_banks'])
    minus=encode(x,p@torch.matrix_exp(-eps*h),state['local_banks'])
    observed=(objective(x,plus)-objective(x,minus))/(2*eps)
    assert torch.equal(plus['labels'],minus['labels'])
    err=abs(expected-observed)/max(abs(expected),abs(observed),1e-12)
    assert err<1e-6
    failed,history=fit(x,g,k,r,2,max_sweeps=150)
    result=dict(analytic_derivative=expected,finite_difference=observed,relative_error=err,
                stalled_loss=history[-1],stalled_differential_check=stationarity(x,failed),
                scope='Fixed assignment envelope derivative; discrete/global recovery is separate.')
    with Path(__file__).with_name('SHARED_LOCAL_GRADIENT_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
