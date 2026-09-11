"""L-BFGS-B in balanced blocks with exact variable-projection evaluations.

Does not infer convergence from a solver success flag. The caller owns time,
source bindings, persisted checkpoints, and native output/program validation.
"""
import time
import numpy as np
import torch
from scipy.optimize import minimize
from projected_sparse_dictionary_v1 import value_gradient


class FitStop(Exception):pass


class ProjectedObjective:
    def __init__(self,native,whitener,raw,indices,values,row_scale,total):
        self.native=native;self.whitener=whitener;self.indices=indices
        self.row_scale=row_scale;self.total=total;self.raw_shape=raw.shape;self.value_shape=values.shape
        self.n_raw=raw.numel();self.scales=(float(raw.norm()),float(values.norm()))
        assert min(self.scales)>0
        self.initial=torch.cat((raw.flatten()/self.scales[0],values.flatten()/self.scales[1])).cpu().numpy().copy()
        self.last=None;self.evaluations=0;self.evaluation_seconds=[]

    def unpack(self,point):
        p=torch.as_tensor(point,dtype=self.native[0].dtype,device=self.native[0].device)
        return (p[:self.n_raw].reshape(self.raw_shape)*self.scales[0],
                p[self.n_raw:].reshape(self.value_shape)*self.scales[1])

    def evaluate(self,point):
        tic=time.perf_counter();raw,values=self.unpack(point)
        loss,gradient,writer,details=value_gradient(self.native,self.whitener,raw,self.indices,values,self.row_scale,self.total)
        assert details['writer_solve']['normal_residual']<=1e-8
        grad=torch.cat((gradient[0].flatten()*self.scales[0],gradient[1].flatten()*self.scales[1]))
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(grad).all())
        value=float(loss);array=grad.cpu().numpy().copy()
        seconds=time.perf_counter()-tic;self.evaluations+=1;self.evaluation_seconds.append(seconds)
        self.last=dict(point=np.array(point,copy=True),value=value,gradient=array,writer=writer,details=details)
        return value,array


def fit(objective,seconds=3600,max_iterations=2500,checkpoint=None):
    started=time.perf_counter();initial_value,_=objective.evaluate(objective.initial)
    initial_details=objective.last['details'];accepted=objective.initial.copy();history=[]
    converged=False;stop='solver_return';last_save=started
    def callback(point):
        nonlocal accepted,converged,stop,last_save
        accepted=np.array(point,copy=True)
        if not np.array_equal(point,objective.last['point']):objective.evaluate(point)
        current=objective.last;details=current['details']
        relative=max(details['dictionary_relative_stationarity'],details['codes_relative_stationarity'])
        change=None if not history else abs(current['value']-history[-1]['loss'])/max(abs(current['value']),1e-12)
        row=dict(iteration=len(history)+1,seconds=time.perf_counter()-started,loss=current['value'],
            relative_stationarity=relative,relative_objective_change=change,details=details,evaluations=objective.evaluations)
        history.append(row)
        converged=len(history)>=2 and all(h['relative_stationarity']<=1e-5 for h in history[-2:]) and change<=1e-8
        if checkpoint and (time.perf_counter()-last_save>=60 or converged):
            checkpoint(accepted,current,history);last_save=time.perf_counter()
        if len(history)==1 or len(history)%5==0 or converged:print(__import__('json').dumps(row),flush=True)
        if converged:stop='joint_convergence';raise FitStop()
        if time.perf_counter()-started>=seconds:stop='time_limit';raise FitStop()
    try:
        result=minimize(objective.evaluate,objective.initial,jac=True,method='L-BFGS-B',callback=callback,
            options=dict(maxiter=max_iterations,maxfun=max_iterations*20,maxcor=10,maxls=25,ftol=0.,gtol=1e-12))
        accepted=np.array(result.x,copy=True)
        solver=dict(success=bool(result.success),status=int(result.status),message=str(result.message))
    except FitStop:
        solver=dict(success=converged,status=None,message=stop)
    # Recompute the accepted state, not an unaccepted line-search candidate.
    objective.evaluate(accepted);current=objective.last
    if checkpoint:checkpoint(accepted,current,history)
    report=dict(initial_loss=initial_value,initial_details=initial_details,final_loss=current['value'],
        final_details=current['details'],converged=converged,stop=stop,solver=solver,
        iterations=len(history),evaluations=objective.evaluations,seconds=time.perf_counter()-started,
        evaluation_seconds=objective.evaluation_seconds,history=history,parameter_block_scales=objective.scales,
        maximum_accepted_loss_increase=max([0.]+[history[i]['loss']-(initial_value if i==0 else history[i-1]['loss']) for i in range(len(history))]))
    return accepted,current['writer'],report
