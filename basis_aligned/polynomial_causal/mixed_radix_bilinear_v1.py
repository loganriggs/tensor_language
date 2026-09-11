"""Small mixing stages give compact full-rank bilinear weight programs.

Representability/execution control only. No claim of recovering native weights.
Inspired by butterfly factorization; mixed-radix extension explicitly implemented.
"""
import json
import math
from pathlib import Path
import torch
from torch import nn


class MixedRadix(nn.Module):
    def __init__(self,radices):
        super().__init__();self.radices=tuple(radices);self.dim=math.prod(radices)
        stages=[];stride=1
        for radix in radices:
            outer=self.dim//(radix*stride)
            weights=torch.eye(radix).expand(outer,stride,radix,radix).clone()
            weights+=.15*torch.randn_like(weights)
            stages.append(nn.Parameter(weights));stride*=radix
        self.stages=nn.ParameterList(stages)

    def forward(self,x):
        stride=1
        for radix,weight in zip(self.radices,self.stages):
            outer=self.dim//(radix*stride)
            x=x.reshape(*x.shape[:-1],outer,radix,stride)
            x=torch.einsum('...ors,osqr->...oqs',x,weight).reshape(*x.shape[:-3],self.dim)
            stride*=radix
        return x

    def matrix(self):
        return self(torch.eye(self.dim,dtype=self.stages[0].dtype,device=self.stages[0].device)).T


def price(radices,branches=1):
    d=math.prod(radices)
    maps=3*branches
    return dict(dim=d,branches=branches,maps=maps,
        learned_coefficients=maps*d*sum(radices),
        dense_factor_coefficients=maps*d*d,
        quadratic_products=branches*d,
        multiplications=maps*d*sum(radices)+branches*d,
        additions=maps*d*sum(r-1 for r in radices)+(branches-1)*d,
        scope='Bilinear component only; fixed wiring and code explicit, native unembedding/background retained.')


def control():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(602)
    transforms=[MixedRadix([2,2,2,3]) for _ in range(3)]
    a,c,b=[t.matrix() for t in transforms];d=a.shape[0]
    x=torch.randn(31,d)
    actual=transforms[2](transforms[0](x)*transforms[1](x))
    tensor=torch.einsum('ok,ki,kj->oij',b,a,c)
    tensor=(tensor+tensor.transpose(-1,-2))/2
    dense=torch.einsum('ni,oij,nj->no',x,tensor,x)
    error=float((actual-dense).norm()/dense.norm())
    spectra=[torch.linalg.svdvals(tensor.movedim(axis,0).reshape(d,-1)) for axis in range(3)]
    ranks=[int(torch.linalg.matrix_rank(tensor.movedim(axis,0).reshape(d,-1))) for axis in range(3)]
    margins=[float(s[-1]/s[0]) for s in spectra]
    actual_count=sum(p.numel() for t in transforms for p in t.parameters())
    count=price([2,2,2,3]);assert actual_count==count['learned_coefficients']
    # Differential replay: sparse stage execution and its dense matrix form.
    params=[p for t in transforms for p in t.parameters()]
    g1=torch.autograd.grad(actual.square().mean(),params,retain_graph=True)
    g2=torch.autograd.grad(dense.square().mean(),params)
    grad_error=float((sum((u-v).square().sum() for u,v in zip(g1,g2))/sum(u.square().sum() for u in g1)).sqrt())
    result=dict(predictions={'pred_a_execution':max(error,grad_error)<=1e-10,
                            'pred_b_full_mode_ranks':ranks==[d]*3 and min(margins)>1e-10,
                            'pred_c_compact':actual_count<count['dense_factor_coefficients']},
        execution_relative_error=error,gradient_relative_error=grad_error,
        unfolding_ranks=ranks,minimum_singular_ratios=margins,toy_price=count,
        native_dimension_price=price([2]*7+[3,3],branches=4),
        body_forwards=0,corpus_access=False,gpu_access=False,
        scope='Planted random construction, no fitting. Full-rank compactness is possible; no evidence yet that native weights admit this wiring or can be optimized into it.')
    with Path(__file__).with_name('MIXED_RADIX_BILINEAR_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert all(result['predictions'].values())


if __name__=='__main__':control()
