"""Dense control for shared-square gather/scatter and reduced output solve."""
import json
from pathlib import Path
import numpy as np
import torch
from shared_square_ll1_v1 import Objective,canonicalize,execute
from symmetric_ll1_projected_v1 import output_solve
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2);torch.manual_seed(2301)
    target=(torch.randn(11,9),torch.randn(11,9),torch.randn(7,11));truth=dense(*target);total=truth.square().sum()
    readers=torch.randn(5,9);indices=torch.tensor([[0,1,2],[0,3,4],[1,3,4]])
    s=torch.randn(3,3);objective=Objective(target,readers,indices,s,total,.01)
    point=objective.initial;value,gradient=objective.evaluate(point)
    packed=torch.tensor(point,requires_grad=True)
    rr,ss=[v.reshape(shape)*scale for v,shape,scale in zip(packed.split(objective.sizes),objective.shapes,objective.scales)]
    rr=rr/rr.norm(dim=-1,keepdim=True);ss=ss/ss.norm(dim=-1,keepdim=True)
    a=rr[indices];q=torch.einsum('gk,gki,gkj->gij',ss,a,a)
    gram=torch.einsum('gij,hij->gh',q,q);rhs=torch.einsum('oij,gij->go',truth,q)
    c=torch.linalg.solve(gram+.01*torch.diag(gram.diag()),rhs)
    fitted=torch.einsum('go,gij->oij',c,q)
    loss=((truth-fitted).square().sum()+.01*(c.square().sum(1)*q.square().sum((1,2))).sum())/total
    actual=torch.autograd.grad(loss,packed)[0].detach().numpy()
    grad_error=float(np.linalg.norm(gradient-actual)/np.linalg.norm(actual))
    direction=np.random.default_rng(2301).normal(size=len(point));direction/=np.linalg.norm(direction);h=1e-6
    fd=(objective.evaluate(point+h*direction)[0]-objective.evaluate(point-h*direction)[0])/(2*h)
    fd_error=float(abs(fd-gradient@direction)/max(1.,abs(fd),abs(gradient@direction)))
    rr,ss=objective.physical(point);writer,*_=output_solve(target,rr[indices],ss,.01)
    x=torch.randn(13,9);reference=torch.einsum('oij,ni,nj->no',dense(*cp(rr[indices],ss,writer)),x,x)
    replay=float((execute(rr,indices,ss,writer,x)-reference).norm()/reference.norm())
    canonical=canonicalize(rr[indices],ss,writer)
    canonical_error=float((dense(*cp(*canonical))-dense(*cp(rr[indices],ss,writer))).norm()/truth.norm())
    result=dict(pred_a=max(grad_error,replay,canonical_error,abs(value-float(loss.detach())))<=1e-9,
                pred_b=fd_error<=1e-6,dense_gradient_error=grad_error,packed_fd_error=fd_error,
                executor_replay=replay,canonicalization_replay=canonical_error,
                unique_readers=5,slot_readers=9,scope='Small dense numerical control; no native graph fit or recovery guarantee.')
    Path(__file__).with_name('SHARED_SQUARE_LL1_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b']


if __name__=='__main__':main()
