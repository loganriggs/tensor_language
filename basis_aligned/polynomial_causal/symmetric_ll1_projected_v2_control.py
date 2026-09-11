"""Control-only repair: serialize the NumPy finite-difference predicate as a Python bool."""
import json
from pathlib import Path
import torch
import numpy as np
from symmetric_ll1_projected_v1 import evaluate, Objective, output_solve, cp
from chunked_bilinear_coefficient_v1 import dense

def control():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(1801)
    target=(torch.randn(11,9),torch.randn(11,9),torch.randn(7,11));truth=dense(*target);total=truth.square().sum()
    a=torch.randn(3,3,9,requires_grad=True);s=torch.randn(3,3,requires_grad=True);penalty=.01
    # Independent dense conditional-output construction, differentiated through the solve.
    q=torch.einsum('gk,gki,gkj->gij',s,a,a);g=torch.einsum('gij,hij->gh',q,q)
    rhs=torch.einsum('oij,gij->og',truth,q);c=torch.linalg.solve(g+penalty*torch.diag(g.diag()),rhs.T)
    prediction=torch.einsum('go,gij->oij',c,q)
    loss=((truth-prediction).square().sum()+penalty*(c.square().sum(1)*q.square().sum((1,2))).sum())/total
    gradient=torch.autograd.grad(loss,(a,s));value,actual,c2,details=evaluate(target,a.detach(),s.detach(),total,penalty)
    errors=[float((x-y).norm()/y.norm()) for x,y in zip(actual,gradient)]
    objective=Objective(target,a.detach(),s.detach(),total,penalty);point=objective.initial;vv,gg=objective.evaluate(point)
    direction=np.random.default_rng(1801).normal(size=len(point));direction/=np.linalg.norm(direction);h=1e-6
    finite=(objective.evaluate(point+h*direction)[0]-objective.evaluate(point-h*direction)[0])/(2*h)
    fd=float(abs(finite-gg@direction)/max(1.,abs(gg@direction)))
    # A signed scaling of each complete Q is absorbed by the output solve.
    aa,ss=objective.physical(point);scaled=ss*torch.tensor([2.,-.3,4.])[:,None]
    value2,_,c3,details3=evaluate(target,aa,scaled,total,penalty)
    cbase,_,_,_=output_solve(target,aa,ss,penalty)
    gauge=float((dense(*cp(aa,ss,cbase))-dense(*cp(aa,scaled,c3))).norm()/truth.norm())
    result=dict(predictions=dict(pred_a_dense=max(errors+[abs(float(value-loss.detach())),float((c2-c).norm()/c.norm())])<=1e-8,
                                 pred_b_normalized_fd=fd<=1e-6,pred_c_scale_gauge=max(abs(float(value2)-vv),gauge)<=1e-8,
                                 pred_d_normal=details['output_normal_residual']<=1e-9),
                gradient_errors=errors,normalized_fd_error=fd,scale_function_error=gauge,details=details,
                scope='Exact conditional group outputs and first reduced gradient. No native projected fit, global convergence, Hessian or identifiability guarantee.')
    Path(__file__).with_name('SYMMETRIC_LL1_PROJECTED_V2_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert all(result['predictions'].values())

if __name__=="__main__":control()
