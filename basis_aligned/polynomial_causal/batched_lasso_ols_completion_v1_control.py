"""Batched completion versus the already controlled scalar implementation."""
import json
from pathlib import Path
import torch
from lasso_ols_completion_v1 import encode as scalar
from batched_lasso_ols_completion_v1 import encode as batched

@torch.no_grad()
def compare(device='cpu'):
    gen=torch.Generator().manual_seed(843);rows=[]
    for name in ('orthogonal','rectangular'):
        if name=='orthogonal':
            basis=torch.eye(4);x=torch.tensor([[.8,.4,.3,.2],[.2,.3,.4,.8],[1.,0.,0.,0.]])
            k,penalty=2,.5
        else:
            basis=torch.randn(20,12,generator=gen);basis/=basis.norm(dim=1,keepdim=True)
            x=torch.randn(32,12,generator=gen);k,penalty=6,.2
        basis=basis.to(device=device,dtype=torch.float64);x=x.to(device=device,dtype=torch.float64)
        si,sv,sr=scalar(basis,x,k,penalty);bi,bv,br=batched(basis,x,k,penalty,batch_size=5)
        dense_s=torch.zeros(len(x),len(basis),device=device,dtype=x.dtype).scatter_(1,si,sv)
        dense_b=torch.zeros_like(dense_s).scatter_(1,bi,bv)
        error=float(((dense_s-dense_b)@basis).norm()/(dense_s@basis).norm())
        same=all(set(a)==set(b) for a,b in zip(si.cpu().tolist(),bi.cpu().tolist()))
        rows.append(dict(name=name,physical_error=error,same_supports=same,
            normal_residual=br['support_ls_normal_residual'],raw_converged=sr['converged'] and br['converged'],
            initial_support_counts=br['initial_support_counts']))
    passed=all(r['physical_error']<=1e-9 and r['same_supports'] and r['normal_residual']<=1e-9 and r['raw_converged'] for r in rows)
    return dict(passed=passed,rows=rows,device=device,
                scope='Same support-completion algorithm, CPU/GPU device as recorded; not native quality or runtime evidence.')

if __name__=='__main__':
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    result=compare()
    with Path(__file__).with_name('BATCHED_LASSO_OLS_COMPLETION_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True);assert result['passed']
