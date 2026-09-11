"""Exact elimination of all LL1 output vectors in the whole-group objective."""
import json
from pathlib import Path
import torch
import numpy as np
from symmetric_ll1_objective_v1 import value_gradient,cp
from chunked_bilinear_coefficient_v1 import dense


def output_solve(target,a,s,penalty=.01):
    m,r,d=a.shape
    flat=a.reshape(m*r,d)
    products=(flat@flat.T).square().reshape(m,r,m,r)
    gram=torch.einsum('irjs,ir,js->ij',products,s,s)
    left,right,writers=target
    projections=((left@flat.T)*(right@flat.T)).reshape(len(left),m,r)
    cross=(projections*s[None,:,:]).sum(2)
    rhs=writers@cross
    system=gram+penalty*torch.diag(gram.diag())
    c=torch.linalg.solve(system,rhs.T)
    return c,gram,rhs,system


def evaluate(target,a,s,total,penalty=.01):
    with torch.no_grad():
        c,gram,rhs,system=output_solve(target,a,s,penalty)
        loss,(ga,gs,gc),details=value_gradient(target,a,s,c,total,penalty)
        details['output_normal_residual']=float((system@c-rhs.T).norm()/rhs.norm().clamp_min(1e-300))
        details['output_stationarity']=float(gc.norm())
        details['reduced_identity_error']=abs(float(loss-(1-(rhs.T*c).sum()/total)))
        return loss,(ga,gs),c,details


class Objective:
    def __init__(self,target,a,s,total,penalty=.01):
        self.target,self.total,self.penalty=target,total,penalty;self.device=a.device
        self.shapes=[a.shape,s.shape];self.sizes=[a.numel(),s.numel()]
        norm=a.norm(dim=-1,keepdim=True);a=a/norm;s=s*norm.squeeze(-1).square();s=s/s.norm(dim=-1,keepdim=True)
        self.scales=[float(a.norm()),float(s.norm())]
        self.initial=torch.cat([(x/scale).flatten().cpu() for x,scale in zip((a,s),self.scales)]).numpy()

    def unpack(self,point):
        return tuple(x.reshape(shape)*scale for x,shape,scale in zip(torch.as_tensor(point,dtype=torch.float64,device=self.device).split(self.sizes),self.shapes,self.scales))

    def physical(self,point):
        a,s=self.unpack(point);return a/a.norm(dim=-1,keepdim=True),s/s.norm(dim=-1,keepdim=True)

    def evaluate(self,point):
        a,s=self.unpack(point);an=a.norm(dim=-1,keepdim=True);sn=s.norm(dim=-1,keepdim=True);aa,ss=a/an,s/sn
        loss,(ga,gs),c,details=evaluate(self.target,aa,ss,self.total,self.penalty)
        ga=(ga-aa*(aa*ga).sum(-1,keepdim=True))/an
        gs=(gs-ss*(ss*gs).sum(-1,keepdim=True))/sn
        self.last=details;self.writer=c
        return float(loss),torch.cat([(g*scale).flatten().cpu() for g,scale in zip((ga,gs),self.scales)]).numpy()


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
    fd=abs(finite-gg@direction)/max(1.,abs(gg@direction))
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
    Path(__file__).with_name('SYMMETRIC_LL1_PROJECTED_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert all(result['predictions'].values())


if __name__=='__main__':control()
