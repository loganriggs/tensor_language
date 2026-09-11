"""An exact shared linear parent can be invisible to eigen-square merging."""
import json
from pathlib import Path
import torch
from shared_square_ll1_v1 import canonicalize,merge_readers,execute
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    u,v,w=torch.eye(3)
    a=torch.stack((torch.stack((u+v,u-v)),torch.stack((u+w,u-w))))/2**.5
    s=torch.tensor([[.5,-.5],[.5,-.5]]);c=torch.eye(2)
    aa,ss,cc=canonicalize(a,s,c)
    readers,indices,weights,clusters=merge_readers(aa,ss,cc,.995)
    torch.manual_seed(2303);x=torch.randn(31,3)
    reference=torch.stack((x[:,0]*x[:,1],x[:,0]*x[:,2]),dim=1)
    error=float((execute(readers,indices,weights,cc,x)-reference).norm()/reference.norm())
    target=torch.stack(((u[:,None]*v[None]+v[:,None]*u[None])/2,
                        (u[:,None]*w[None]+w[:,None]*u[None])/2))
    coefficient=float((dense(*cp(aa,ss,cc))-target).norm()/target.norm())
    cross=float((aa[0]@aa[1].T).abs().max())
    result=dict(pred_a=max(error,coefficient)<=1e-12,pred_b=len(readers)==4 and abs(cross-.5)<=1e-12,
                executor_replay=error,coefficient_replay=coefficient,maximum_cross_group_eigenreader_cosine=cross,
                canonical_square_readers=len(readers),canonical_square_products=4,explicit_linear_readers=3,explicit_mixed_products=2,
                scope='Exact planted counterexample to near-identical eigen-square merging as a complete reuse detector. It does not establish that this particular mixed-parent structure is present in the native weights.')
    Path(__file__).with_name('SHARED_SQUARE_LL1_MIXED_PARENT_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b']


if __name__=='__main__':main()
