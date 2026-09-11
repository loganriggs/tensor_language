"""A separable polynomial can have high-rank feature-removal computations.

Same F in two coordinate systems; no native repair or sparse-price equivalence.
"""
import json
from pathlib import Path
import torch
from scipy.linalg import hadamard
from sparse_reader_program_v1 import SparseReaderProgram


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    n=128;eye=torch.eye(n);basis=torch.tensor(hadamard(n),dtype=torch.float64)/n**.5
    ids=torch.arange(n)[None,:].expand(2*n,-1).clone()
    values=torch.cat((basis.T,basis.T));program=SparseReaderProgram(basis,ids,values,eye)
    x=torch.randn(8,n,generator=torch.Generator().manual_seed(1041))
    replay=float((program(x)-x.square()).norm()/x.square().norm())
    rows=[]
    for name,a,m in [('identity',eye[0],eye[0,:,None]*eye[0,None,:]),
                     ('rotated',basis[0],2*torch.diag(basis[0])-basis[0].square()[:,None]*basis[0,None,:])]:
        transformed=m/2**.5+(1-2**-.5)*(m@a)[:,None]*a[None,:]
        s=torch.linalg.svdvals(transformed);cumulative=s.square().cumsum(0)/s.square().sum()
        rows.append(dict(name=name,rank1=float(cumulative[0]),rank16=float(cumulative[15]),
                         rank90=int(torch.searchsorted(cumulative,torch.tensor(.9)))+1))
        if name=='rotated':
            actual=program(x)-program.remove_features(x,[0]);expected=(x@a)[:,None]*(x@m.T)
            removal_replay=float((actual-expected).norm()/actual.norm())
    passed=dict(pred_a_replay=max(replay,removal_replay)<=1e-10,
                pred_b_simple=abs(rows[0]['rank1']-1)<=1e-12,
                pred_c_mixed=rows[1]['rank16']<.2 and rows[1]['rank90']>=100)
    result=dict(predictions=passed,rows=rows,function_replay=replay,removal_replay=removal_replay,
        scope='Exact diagonal-square function, coordinate counterexample. Dense rotated codes cost more '
              'than eliminating identity zero entries. Does not prove native recovery or same-price repair.')
    with Path(__file__).with_name('FEATURE_REMOVAL_COORDINATE_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert all(passed.values())


if __name__=='__main__':main()
