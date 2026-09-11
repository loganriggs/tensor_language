"""Source influence operator and a rank-r touch bound for a two-QK numerator."""
import json
from pathlib import Path
import torch
from joint_qk_source_ports_v1 import low_rank_norm,source_ports
from joint_router_polynomial_gram_v1 import dense_biquadratic


def influence(q1,q2,k1,k2):
    """Small source coordinates recommended; returns S with trace(S)=||T||²."""
    metric=torch.eye(k1.shape[1],dtype=k1.dtype,device=k1.device,requires_grad=True)
    symmetric=(metric+metric.T)/2
    value=low_rank_norm(q1@q1.T,q2@q2.T,q1@q2.T,
                        k1@symmetric@k1.T,k2@symmetric@k2.T,k1@symmetric@k2.T)
    derivative=torch.autograd.grad(value,metric)[0]
    return ((derivative+derivative.T)/4).detach()


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1231)
    q1,q2=torch.randn(3,5),torch.randn(3,5)
    k1,k2=torch.randn(3,7),torch.randn(3,7)
    tensor=dense_biquadratic(q1.T@k1,q2.T@k2)
    expected=torch.einsum('ijac,ijbc->ab',tensor,tensor)
    observed=influence(q1,q2,k1,k2)
    error=float((observed-expected).norm()/expected.norm())
    eig,vectors=torch.linalg.eigh(observed);total=tensor.square().sum()
    trace_error=float(abs(observed.trace()/total-1));negative=max(0.,-float(eig[0]/total))
    rank=2;basis=vectors[:,-rank:];ports=source_ports(q1,q2,k1,k2,basis)
    touch=ports['inside']+ports['mixed'];identity_error=float(abs(touch-(2*torch.trace(basis.T@observed@basis)-ports['inside']))/total)
    bound=min(1.,float(2*eig[-rank:].sum()/total));lower=float(touch/total)
    out=dict(instrument_passed=max(error,trace_error,negative,identity_error,max(0.,lower-bound))<1e-10,
             dense_influence_relative_error=error,trace_relative_error=trace_error,negative_eigenvalue_fraction=negative,
             touch_identity_relative_error=identity_error,toy_rank2_spectral_touch=lower,toy_rank2_upper_bound=bound,
             scope='Coefficient-only influence bound for any rank-r source subspace of a separately symmetric biquadratic numerator. Spectral subspace is a constructive candidate, not a globally optimal touch projector; native runs pending.')
    Path(__file__).with_name('JOINT_QK_SOURCE_INFLUENCE_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));assert out['instrument_passed']


if __name__=='__main__':control()
