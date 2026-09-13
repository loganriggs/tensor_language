"""Explicit 2!*3! symmetrization verifies the closed contraction independently."""
import itertools,json
from pathlib import Path
import torch
from joint_qk_source_gram_v1 import gram


def main():
    torch.set_num_threads(2);torch.manual_seed(9581)
    cases=[]
    for nq,ns,m in [(3,4,2),(4,3,3),(1,2,2)]:
        a,b=[torch.randn(nq,ns,dtype=torch.float64) for _ in range(2)]
        f=torch.randn(m,ns,dtype=torch.float64)
        raw=torch.einsum('pi,qj,ok->opqijk',a,b,f)
        symmetric=torch.zeros_like(raw)
        for qp in [(1,2),(2,1)]:
            for sp in itertools.permutations([3,4,5]):
                symmetric+=raw.permute(0,*qp,*sp)/12
        direct=symmetric.flatten(1)@symmetric.flatten(1).T
        predicted,terms=gram(a,b,f)
        error=float((predicted-direct).norm()/direct.norm())
        traced=torch.einsum('oppijk->oijk',symmetric).flatten(1)
        trace_error=float((terms['query_trace']-traced@traced.T).norm()/(traced@traced.T).norm())
        assert max(error,trace_error)<1e-10
        q=torch.randn(nq,dtype=torch.float64);s=torch.randn(ns,dtype=torch.float64)
        value=torch.einsum('opqijk,p,q,i,j,k->o',symmetric,q,q,s,s,s)
        exact=(q@a@s)*(q@b@s)*(f@s)
        function_error=float((value-exact).norm()/exact.norm());assert function_error<1e-10
        cases.append(dict(shape=[nq,ns,m],coefficient_gram_error=error,query_trace_error=trace_error,
                          function_error=function_error,
                          naive_half_gaussian_relative_error=float((terms['gaussian']/2-direct).norm()/direct.norm())))
    with Path(__file__).with_name('JOINT_QK_SOURCE_GRAM_V1_CONTROL.json').open('x') as out:
        json.dump(cases,out,indent=2);out.write('\n')
    print(json.dumps(cases,indent=2))


if __name__=='__main__':main()
