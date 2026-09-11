"""A selected coefficients B all factor gradients C fixed-objective replay <=1e-10.
No optimizer criterion relaxation. Native speed measured in managed continuation.
"""
from pathlib import Path
import json,torch
from coupled_sparse_path_v1 import coefficients
from coupled_sparse_path_v2 import selected_coefficients


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(11501)
    out=Path(__file__).with_name('COUPLED_SPARSE_PATH_V2_CONTROL.json');assert not out.exists();rows=[]
    for mode,count in [('joint',2),('independent',4)]:
        for orthogonal in (False,True):
            l=torch.randn(2,13,7,requires_grad=True);r=torch.randn_like(l,requires_grad=True);w=torch.randn(5,13,requires_grad=True)
            bank=torch.randn(count,7,3)
            if orthogonal:bank=torch.linalg.qr(bank).Q
            bank=bank.detach().requires_grad_();args=(l,r,w,bank)
            full=coefficients(*args,mode);support=torch.randperm(full.shape[1])[:11]
            old=full[:,support];new=selected_coefficients(*args,mode,support)
            go=torch.autograd.grad(old.square().sum(),args);gn=torch.autograd.grad(new.square().sum(),args)
            errors=[float(((a-b).norm()/b.norm()).detach()) for a,b in zip(gn,go)]
            rows.append(dict(mode=mode,orthogonal=orthogonal,coefficient_error=float(((new-old).norm()/old.norm()).detach()),gradient_errors=errors,
                relative_loss_error=float(abs(new.square().sum()/old.square().sum()-1).detach())))
    result=dict(pred_a=max(r['coefficient_error'] for r in rows)<=1e-10,pred_b=max(max(r['gradient_errors']) for r in rows)<=1e-10,
        pred_c=max(r['relative_loss_error'] for r in rows)<=1e-10,rows=rows,scope='Same full-U sparse path fixed-support objective/gradients, active-column evaluation only; no convergence or native speed claim yet.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b'] and result['pred_c']

if __name__=='__main__':main()
