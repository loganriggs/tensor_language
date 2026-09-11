"""Exact conditional rank-r partner objective for selecting the shared input reader."""
import json
from pathlib import Path
import torch
from shared_input_factor_v1 import native_partner,value_gradient
from congruence_block_operator_v1 import sandwich


def objective(raw,left,right,down,output_root,total,rank):
    a=raw/raw.norm()
    m=native_partner(a,left,right,down)
    j=torch.eye(len(a),dtype=a.dtype,device=a.device)+(2**.5-1)*a[:,None]*a[None,:]
    singular=torch.linalg.svdvals(output_root.T@m@j)
    return .5*singular[:rank].square().sum()/total


def value_and_gradient(a,left,right,down,output_root,total,rank):
    raw=a.detach().clone().requires_grad_()
    value=objective(raw,left,right,down,output_root,total,rank)
    gradient=torch.autograd.grad(value,raw)[0]
    return value.detach(),gradient.detach()


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1151)
    l,r,d,u=torch.randn(9,7),torch.randn(9,7),torch.randn(5,9),torch.randn(11,5)
    root=torch.linalg.cholesky(u.T@u);k=(u@d).T@(u@d);s=sandwich(l,r,k,torch.eye(7));total=s.trace()
    a=torch.randn(7);a/=a.norm();rank=2
    score,gradient=value_and_gradient(a,l,r,d,root,total,rank)
    m=native_partner(a,l,r,d);j=torch.eye(7)+(2**.5-1)*a[:,None]*a[None,:]
    invj=torch.eye(7)+(2**-.5-1)*a[:,None]*a[None,:]
    left,values,right=torch.linalg.svd(root.T@m@j,full_matrices=False)
    approximation=(left[:,:rank]*values[:rank])@right[:rank]
    mhat=torch.linalg.solve(root.T,approximation)@invj
    native_h=(l[:,:,None]*r[:,None,:]+r[:,:,None]*l[:,None,:])/2
    target=torch.einsum('vk,kij->vij',u@d,native_h)
    b=u@mhat;candidate=(a[None,:,None]*b[:,None,:]+b[:,:,None]*a[None,None,:])/2
    dense_capture=1-(target-candidate).square().sum()/target.square().sum()
    direct_error=float(abs(score-dense_capture))
    direction=torch.randn(7);direction-=a*(a@direction);direction/=direction.norm()
    step=1e-6
    finite=(objective(a+step*direction,l,r,d,root,total,rank)-objective(a-step*direction,l,r,d,root,total,rank))/(2*step)
    gradient_error=float(abs(finite-gradient@direction))
    unconstrained,_=value_gradient(a,l,r,k,s)
    full_error=float(abs(objective(a,l,r,d,root,total,5)-unconstrained/total))
    result=dict(instrument_passed=max(direct_error,full_error)<1e-10 and gradient_error<1e-6,
                dense_native_coefficient_capture_error=direct_error,tangent_finite_difference_error=gradient_error,
                unrestricted_partner_bridge_error=full_error,
                cutoff_singular_gap=float(values[rank-1]-values[rank]),
                scope='Exact fixed-reader rank constraint and differentiable variable-projection objective; generic nonzero cutoff gap control. No native joint optimization or global guarantee.')
    Path(__file__).with_name('SHARED_INPUT_RANKED_PARTNER_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
