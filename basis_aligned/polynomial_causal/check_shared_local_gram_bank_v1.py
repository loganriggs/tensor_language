"""Leading function projections match direct SVD for both Gram orientations."""
import json
from pathlib import Path
import torch
from shared_local_subspaces_v1 import bank
from shared_local_gram_bank_v1 import gram_bank


def controls(device='cpu'):
    torch.manual_seed(9517)
    cases=[]
    for n,d,rank,deficient in [(81,32,8,False),(21,64,8,False),(3,12,5,False),(30,20,8,True)]:
        x=torch.randn(n,d,dtype=torch.float64,device=device)
        if deficient:x=x[:,:3]@torch.randn(3,d,dtype=x.dtype,device=device)
        p,q=bank(x,rank),gram_bank(x,rank)
        a,b=(x@p.T)@p,(x@q.T)@q
        err=float((a-b).norm()/x.norm())
        orth=float((q@q.T-torch.eye(len(q),device=device,dtype=q.dtype)).norm())
        assert max(err,orth)<1e-10
        cases.append(dict(shape=[n,d],rank=rank,deficient=deficient,
                          prediction_relative_error=err,orthogonality_error=orth))
    return cases


if __name__=='__main__':
    torch.set_num_threads(2)
    result=controls()
    with Path(__file__).with_name('SHARED_LOCAL_GRAM_BANK_V1_CONTROL.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
