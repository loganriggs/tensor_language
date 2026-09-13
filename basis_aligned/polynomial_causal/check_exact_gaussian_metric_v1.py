"""Exact independent Gaussian query/source average, unnormalized current values.

Finite-query bound red-team: predict exact generalized distortion <=1.2.
Tiny quadrature validates the query Wick contraction before using real weights.
"""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from shared_key_value_moment_v1 import moment
from head17_source_interface_v1 import CHECKPOINT


def average(f,a,b):
    # a,b: source x query; f: output x source. Query covariance identity.
    aa,bb=a.T@a,b.T@b
    s=(a.T@b+b.T@a)/2
    c=torch.trace(aa)*torch.trace(bb)+2*(aa*bb).sum()+2*torch.trace(s)**2+4*s.square().sum()
    fa,fb=f@a,f@b
    h0=c*(f@f.T)
    correction=(2*torch.trace(bb)*(fa@fa.T)+4*fa@bb@fa.T
                +2*torch.trace(aa)*(fb@fb.T)+4*fb@aa@fb.T
                +4*torch.trace(s)*(fa@fb.T+fb@fa.T)
                +8*(fa@s@fb.T+fb@s@fa.T))
    return h0,h0+correction


def main():
    torch.set_num_threads(2)
    torch.manual_seed(7131257)
    f=torch.randn(3,5,dtype=torch.float64)
    a,b=[torch.randn(5,3,dtype=torch.float64) for _ in range(2)]
    nodes,weights=np.polynomial.hermite.hermgauss(3)
    quadrature=torch.zeros(3,3,dtype=torch.float64)
    for ii in itertools.product(range(3),repeat=3):
        q=torch.tensor([2**.5*nodes[i] for i in ii],dtype=torch.float64)
        weight=np.prod([weights[i]/np.pi**.5 for i in ii])
        quadrature+=float(weight)*moment(f,a@q,b@q)
    _,exact=average(f,a,b)
    error=float((exact-quadrature).norm()/exact.norm())
    assert error<1e-12
    sd=torch.load(CHECKPOINT,weights_only=True,map_location='cpu',mmap=True)
    def head(name):
        return sd[f'transformer.h.17.attn.{name}.weight'].reshape(9,128,1152)[2].double()
    a,b=head('c_k').T@head('c_q'),head('c_k2').T@head('c_q2')
    f=(1-float(sd['transformer.h.17.attn.lamb']))*head('c_v')
    h0,h=average(f,a,b)
    l=torch.linalg.cholesky(h0)
    x=torch.linalg.solve_triangular(l,h,upper=False)
    white=torch.linalg.solve_triangular(l,x.T,upper=False).T
    ev=torch.linalg.eigvalsh((white+white.T)/2)
    low,high=float(ev[0]),float(ev[-1])
    assert low>0
    result=dict(pred_a=high/low<=1.2,quadrature_relative_error=error,
                lambda_min=low,lambda_max=high,squared_objective_distortion=high/low,
                error_norm_factor=(high/low)**.5,
                scope='Exact Gaussian query/source average via fourth/sixth moments; no query sampling. Unnormalized current-value only, independent query/source, zero rotary displacement. No retained gates, inherited values, native input distribution or behavioral guarantee.')
    with Path(__file__).with_name('EXACT_GAUSSIAN_METRIC_V1_RESULT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
