"""Damped joint GN with exact conditional writer projection after trial steps."""
import time
import torch
from joint_quadratic_fit_v1 import product_cross
from symmetric_product_als_v1 import pcg
from symmetric_product_gauss_newton_v1 import pack,unpack,normal_action,normal_diagonal,loss_gradient


def canonical_project(parameters,target,penalty):
    a,b,z=parameters
    a=a/a.norm(dim=1,keepdim=True).clamp_min(1e-30)
    b=b/b.norm(dim=1,keepdim=True).clamp_min(1e-30)
    l,r,d=target;gram=product_cross(a,b,a,b)
    cross=d@product_cross(l,r,a,b)
    z=torch.linalg.solve(gram+penalty*torch.diag(gram.diag()),cross.T).T
    return a,b,z


def fit(parameters,target,total,penalty=.01,seconds=120,max_steps=10000):
    parameters=canonical_project(parameters,target,penalty);mu=.01
    history=[];iterations=0;accepted=0;rejected=0;inexact=0;start=time.perf_counter()
    objective,g,terms=loss_gradient(parameters,target,total,penalty)
    initial=float(objective);reason="budget_limit"
    while time.perf_counter()-start<seconds and len(history)<max_steps:
        diagonal=normal_diagonal(parameters,penalty)/total
        diagonal=diagonal.clamp_min(diagonal.max()*1e-12)
        action=lambda v:normal_action(parameters,v,penalty)/total
        step,cg=pcg(lambda v:action(v)+mu*diagonal*v,-g,lambda v:v/((1+mu)*diagonal),tolerance=1e-3,max_iterations=100)
        iterations+=cg['iterations'];inexact+=not cg['converged']
        predicted=float(-2*(g*step).sum()-(step*action(step)).sum())
        if 0<=predicted<=64*torch.finfo(g.dtype).eps*max(abs(float(objective)),1.):
            reason="predicted_decrease_below_numerical_resolution";break
        candidate=unpack(pack(parameters)+step,parameters)
        candidate=canonical_project(candidate,target,penalty)
        newobj,newg,newterms=loss_gradient(candidate,target,total,penalty)
        actual=float(objective-newobj);rho=actual/predicted if predicted>0 else float('-inf')
        ok=bool(torch.isfinite(newobj)) and predicted>0 and actual>0 and rho>1e-4
        if ok:
            parameters=candidate;objective,g,terms=newobj,newg,newterms;accepted+=1
            if rho>.75:mu=max(mu/3,1e-8)
            elif rho<.25:mu=min(mu*2,1e8)
        else:mu=min(mu*10,1e8);rejected+=1
        row=dict(step=len(history)+1,objective=float(objective),reconstruction=float(terms['reconstruction']),half_gradient_norm=float(g.norm()),accepted=ok,damping=mu,gain_ratio=rho,predicted_decrease=predicted,actual_decrease=actual,inner_iterations=cg['iterations'],inner_true_relative_residual=cg['true_relative_residual'])
        history.append(row)
        if len(history)%10==0:print(__import__('json').dumps(row),flush=True)
        if float(g.norm())<1e-14:break
    return parameters,dict(initial=initial,final=float(objective),history=history,accepted=accepted,rejected=rejected,inner_iterations=iterations,inexact_inner_solves=inexact,fit_seconds=time.perf_counter()-start,terminal_reason=reason)
