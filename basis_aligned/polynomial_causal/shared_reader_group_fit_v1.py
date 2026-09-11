"""Normalized reader/partner coordinates for the exact shared-group objective."""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from shared_reader_group_objective_v1 import value_gradient


class Objective:
    def __init__(self,target,parts,total,penalty):
        self.target,self.total,self.penalty=target,total,penalty
        self.shapes=[x.shape for x in parts];self.sizes=[x.numel() for x in parts]
        self.device=parts[0].device
        a,v,w=parts
        an=a.norm(dim=-1,keepdim=True);vn=v.norm(dim=-1,keepdim=True)
        parts=(a/an,v/vn,w*an[:,:,None]*vn.transpose(1,2))
        self.scales=[float(x.norm()) for x in parts]
        self.initial=torch.cat([(x/s).flatten().cpu() for x,s in zip(parts,self.scales)]).numpy()

    def unpack(self,point):
        values=torch.as_tensor(point,device=self.device,dtype=torch.float64).split(self.sizes)
        return tuple(v.reshape(shape)*scale for v,shape,scale in zip(values,self.shapes,self.scales))

    def physical(self,point):
        a,v,w=self.unpack(point)
        return a/a.norm(dim=-1,keepdim=True),v/v.norm(dim=-1,keepdim=True),w

    def evaluate(self,point):
        a,v,w=self.unpack(point);an=a.norm(dim=-1,keepdim=True);vn=v.norm(dim=-1,keepdim=True)
        aa,vv=a/an,v/vn
        loss,(ga,gv,gw),details=value_gradient(self.target,aa,vv,w,self.total,self.penalty)
        ga=(ga-aa*(aa*ga).sum(-1,keepdim=True))/an
        gv=(gv-vv*(vv*gv).sum(-1,keepdim=True))/vn
        gradient=torch.cat([(g*s).flatten().cpu() for g,s in zip((ga,gv,gw),self.scales)]).numpy()
        self.last=details
        return float(loss),gradient


def fit(objective,seconds=120,max_iterations=1000):
    start=time.perf_counter();point=objective.initial.copy();history=[]
    initial,_=objective.evaluate(point)
    class TimeLimit(Exception):pass
    def accepted(x):
        nonlocal point
        point=x.copy();value,gradient=objective.evaluate(point)
        history.append(dict(iteration=len(history)+1,seconds=time.perf_counter()-start,loss=value,
                            gradient_inf=float(np.max(np.abs(gradient)))))
        if len(history)%10==0:print(history[-1],flush=True)
        if time.perf_counter()-start>=seconds:raise TimeLimit
    try:
        result=minimize(objective.evaluate,point,jac=True,method='L-BFGS-B',callback=accepted,
                        options=dict(maxiter=max_iterations,maxcor=10,maxls=30,ftol=0.,gtol=1e-9))
        point=result.x.copy();termination=str(result.message)
    except TimeLimit:termination='soft_time_limit'
    final,gradient=objective.evaluate(point)
    sequence=[initial]+[h['loss'] for h in history]+[final]
    return point,dict(initial=initial,final=final,details=objective.last,termination=termination,
                      gradient_inf=float(np.max(np.abs(gradient))),
                      maximum_increase=max([0.]+[b-a for a,b in zip(sequence,sequence[1:])]),
                      seconds=time.perf_counter()-start,history=history,
                      scope='Pilot endpoint; no multi-step joint convergence certificate.')
