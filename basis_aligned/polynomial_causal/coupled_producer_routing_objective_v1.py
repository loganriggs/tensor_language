"""Basis-invariant producer alignment plus routing influence; analytic gradients."""
import json
from pathlib import Path
import torch
from producer_function_overlap_v1 import support_whitener,overlap


def trace_quotient(e, numerator, denominator):
    c=e.T@denominator@e;h=e.T@numerator@e;inverse=torch.linalg.inv(c)
    return torch.trace(inverse@h),2*(numerator@e@inverse-denominator@e@inverse@h@inverse)


def producer_numerator(readers,metric):
    whitening=support_whitener(readers@metric@readers.T)
    supported=metric@readers.T@whitening
    return supported@supported.T,whitening.shape[1]


def objective(e,producer_metric,producer_matrix,producer_rank,influence,influence_ceiling,weight=.5):
    sharing,gs=trace_quotient(e,producer_matrix,producer_metric)
    routing,gr=trace_quotient(e,influence,torch.eye(len(e),dtype=e.dtype,device=e.device))
    return (weight*sharing/producer_rank+(1-weight)*routing/influence_ceiling,
            weight*gs/producer_rank+(1-weight)*gr/influence_ceiling)


def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1387)
    d,r=9,3;m=torch.randn(d,d);g=m@m.T+torch.eye(d)
    a=torch.randn(r,d);n,rank=producer_numerator(a,g)
    z=torch.randn(d,d);s=z@z.T;s/=s.trace();ceiling=torch.linalg.eigvalsh(s)[-r:].sum()
    e=torch.linalg.qr(torch.randn(d,r)).Q;value,grad=objective(e,g,n,rank,s,ceiling)
    variable=e.clone().requires_grad_();v,_=objective(variable,g,n,rank,s,ceiling)
    auto,=torch.autograd.grad(v,variable);analytic_error=float((auto-grad).norm()/auto.norm())
    sharing,_=trace_quotient(e,n,g);expected=overlap(e.T,a,g)['mean_squared_cosine']
    share_error=abs(float(sharing/r)-expected)
    change=torch.tensor([[2.,.2,.1],[.1,1.,.3],[0.,.1,.7]])
    regauged,_=objective(e@change,g,n,rank,s,ceiling);gauge_error=abs(float(value-regauged))
    direction=torch.randn_like(e);direction-=e@(e.T@direction);eps=1e-6
    plus,_=objective(e+eps*direction,g,n,rank,s,ceiling)
    minus,_=objective(e-eps*direction,g,n,rank,s,ceiling)
    finite_error=abs(float((plus-minus)/(2*eps)-(grad*direction).sum()))
    exact,_=trace_quotient(a.T,n,g)
    result=dict(instrument_passed=max(analytic_error,share_error,gauge_error,finite_error,abs(float(exact/r)-1))<1e-8,
        analytic_gradient_error=analytic_error,overlap_identity_error=share_error,
        invertible_basis_invariance_error=gauge_error,tangent_finite_difference_error=finite_error,
        exact_matching_space_score=float(exact/r),
        scope='Controlled objective oracle only. Sharing exact for the function-space metric; routing term is influence surrogate, not actual QK tensor touch. Native manifold optimization not run.')
    with Path(__file__).with_name('COUPLED_PRODUCER_ROUTING_OBJECTIVE_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['instrument_passed']


if __name__=='__main__':control()
