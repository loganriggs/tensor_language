"""Extract explicit reader pairs behind producer-function principal cosines."""
import json
from pathlib import Path
import torch
from producer_function_overlap_v1 import support_whitener


def pairs(a,b,metric):
    wa=support_whitener(a@metric@a.T);wb=support_whitener(b@metric@b.T)
    u,c,vh=torch.linalg.svd(wa.T@(a@metric@b.T)@wb,full_matrices=False)
    left=(a.T@wa@u).T;right=(b.T@wb@vh.T).T
    return left,right,c


def shared_private(left,right,cosine,tolerance=1e-10):
    common=(left+right)/(2*(1+cosine)).sqrt()[:,None]
    alpha=((1+cosine)/2).sqrt()
    beta=((1-cosine).clamp_min(0)/2).sqrt()
    supported=1-cosine>tolerance
    private=torch.zeros_like(common)
    private[supported]=(left[supported]-right[supported])/(2*(1-cosine[supported])).sqrt()[:,None]
    return dict(common=common,private=private,alpha=alpha,beta=beta,private_supported=supported)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    g=torch.eye(4)
    a=torch.tensor([[1.,0.,0.,0.],[0.,0.,1.,0.]])
    b=torch.tensor([[.8,.6,0.,0.],[0.,0.,0.,1.]])
    l,r,c=pairs(a,b,g);split=shared_private(l,r,c)
    lp=split['alpha'][:,None]*split['common']+split['beta'][:,None]*split['private']
    rp=split['alpha'][:,None]*split['common']-split['beta'][:,None]*split['private']
    errors=dict(cosine=float((c-torch.tensor([.8,0.])).abs().max()),
        left_replay=float((lp-l).abs().max()),right_replay=float((rp-r).abs().max()),
        common_norm=float((split['common']@g@split['common'].T-torch.eye(2)).abs().max()),
        common_private_orthogonality=float((split['common']@g@split['private'].T).abs().max()))
    singular=torch.tensor([[1.,1.,0.],[1.,1.,0.],[0.,0.,1.]])
    aa=torch.tensor([[1.,0.,0.]]);bb=torch.tensor([[0.,1.,0.]])
    ll,rr,cc=pairs(aa,bb,singular);ss=shared_private(ll,rr,cc)
    difference=ll-ss['alpha'][:,None]*ss['common']
    null_error=float(abs((difference@singular@difference.T).sum()))
    result=dict(instrument_passed=max(errors.values())<1e-10 and null_error<1e-10,
        errors=errors,exact_shared_nullspace_replay_error=null_error,
        point8_cosine_common_energy_share=float(split['alpha'][0]**2),
        scope='Explicit normalized polynomial function pairs and shared/private decomposition. Approximate common function is not an exact shared factor when cosine<1; no semantic claim.')
    with Path(__file__).with_name('PRODUCER_FUNCTION_PAIRS_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
