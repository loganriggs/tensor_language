"""Check existing manifold LBFGS on the new objective without new optimizer code."""
from pathlib import Path
import json,time,hashlib
import torch
from graded_source_projection_v1 import graded_norms,balanced_loss
from quartic_manifold_lbfgs_v1 import fit
from quartic_manifold_cg_v1 import tangent
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73170);dtype=torch.float64;d=6;k=2;tic=time.perf_counter()
    true=torch.linalg.qr(torch.randn(d,k,dtype=dtype)).Q
    forms=[]
    for _ in range(3):
        a=torch.randn(k,k,dtype=dtype);forms.append(true@((a+a.T)/2)@true.T)
    forms=torch.stack(forms);root=torch.eye(d,dtype=dtype);wg=torch.eye(2,dtype=dtype);full=graded_norms(forms,root,wg)
    initial=torch.linalg.qr(true+.15*torch.randn_like(true)).Q[None,:,:];n=torch.ones(1,1,dtype=dtype)
    def evaluate(b,n,divisor,gradient=False):
        b=b.detach().requires_grad_(gradient)
        with torch.set_grad_enabled(gradient):
            loss=balanced_loss(forms,root,b[0],wg,full)
            if gradient:
                gb=torch.autograd.grad(loss,b)[0];gb,gn=tangent(b,n,gb,torch.zeros_like(n))
                return float(loss.detach()),None,gb.detach(),gn.detach()
            return float(loss),None
    b,n,_,history,status=fit(initial,n,evaluate,1.,max_steps=300,max_seconds=30)
    last=history[-1];orth=float((b.transpose(-1,-2)@b-torch.eye(k)).norm())
    result=dict(status=status,final=last,orthogonality=orth,sphere=float(n.norm()),iterations=len(history),seconds=time.perf_counter()-tic,
                monotone=all(history[i+1]['objective']<=history[i]['objective']+1e-10 for i in range(len(history)-1)),
                optimizer_sha=hashlib.sha256((P/'quartic_manifold_lbfgs_v1.py').read_bytes()).hexdigest(),source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    assert last['objective']<=1e-8 and last['projected_gradient_norm']<=1e-6 and orth<=1e-10 and result['monotone']
    out=P/'GRADED_LBFGS_ADAPTER_CONTROL_V1_RESULT.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
