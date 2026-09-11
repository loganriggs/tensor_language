"""Independent dense value/gradient and extreme signed scale control."""
import json
from pathlib import Path
import numpy as np
import torch
from equilibrated_ll1_projected_v1 import Objective,output_solve
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(2401)
    target=(torch.randn(13,9),torch.randn(13,9),torch.randn(7,13));truth=dense(*target);total=truth.square().sum()
    a=torch.randn(3,3,9);s=torch.randn(3,3);obj=Objective(target,a,s,total,.01)
    value,gradient=obj.evaluate(obj.initial)
    packed=torch.tensor(obj.initial,requires_grad=True)
    aa,ss=[v.reshape(shape)*scale for v,shape,scale in zip(packed.split(obj.sizes),obj.shapes,obj.scales)]
    aa=aa/aa.norm(dim=-1,keepdim=True);ss=ss/ss.norm(dim=-1,keepdim=True)
    q=torch.einsum('gk,gki,gkj->gij',ss,aa,aa);gram=torch.einsum('gij,hij->gh',q,q)
    rhs=torch.einsum('oij,gij->go',truth,q);c=torch.linalg.solve(gram+.01*torch.diag(gram.diag()),rhs)
    prediction=torch.einsum('go,gij->oij',c,q)
    loss=((prediction-truth).square().sum()+.01*(c.square().sum(1)*q.square().sum((1,2))).sum())/total
    exact=torch.autograd.grad(loss,packed)[0].detach().numpy()
    gradient_error=float(np.linalg.norm(exact-gradient)/np.linalg.norm(exact))
    aa,ss=obj.physical(obj.initial);cb,_,_,base=output_solve(target,aa,ss,.01)
    scaled=ss*torch.tensor([1e-8,-1.,1e8])[:,None];cs,_,_,stats=output_solve(target,aa,scaled,.01)
    scale_replay=float((dense(*cp(aa,ss,cb))-dense(*cp(aa,scaled,cs))).norm()/truth.norm())
    direction=np.random.default_rng(2401).normal(size=len(gradient));direction/=np.linalg.norm(direction);h=1e-6
    finite=(obj.evaluate(obj.initial+h*direction)[0]-obj.evaluate(obj.initial-h*direction)[0])/(2*h)
    fd_error=float(abs(finite-gradient@direction)/max(1.,abs(finite),abs(gradient@direction)))
    result=dict(pred_a=max(gradient_error,abs(value-float(loss.detach())),scale_replay)<1e-9,pred_b=fd_error<1e-6,
                pred_c=stats['output_system_condition']<=stats['output_condition_bound']*(1+1e-10),
                dense_gradient_error=gradient_error,signed_scale_replay=scale_replay,packed_fd_error=fd_error,
                scaled_solve=stats,scope='Algebraically equivalent conditional output solve; no native convergence or global recovery claim.')
    Path(__file__).with_name('EQUILIBRATED_LL1_PROJECTED_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b'] and result['pred_c']


if __name__=='__main__':main()
