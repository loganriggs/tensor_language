"""Bound redundant raw scales and judge convergence in fresh graph coordinates."""
import time
import numpy as np
from scipy.optimize import Bounds,minimize
import torch
from shared_reader_variable_projection_v2 import Objective


def fit(target,graph,total,penalty=.01,seconds=1200,max_cycles=100,iterations_per_cycle=200,
        gradient_bar=1e-7,progress_bar=1e-6):
    started=time.perf_counter();history=[];cycles=[];state=graph;initial=None
    reason='cycle_limit';converged=False;max_bound=0.;max_replay=0.
    for cycle in range(max_cycles):
        objective=Objective(target,state,total,penalty=penalty)
        point=objective.initial.copy()
        before,_=objective.evaluate(point)
        if initial is None:initial=before
        limits=np.concatenate([np.full(size,1/scale) for size,scale in zip(objective.sizes,objective.scales)])
        # Stored graph readers/private bases/writers are normalized, so no
        # substantive projection of the starting function is required.
        assert np.max(np.abs(point)-limits)<=1e-12
        point=np.clip(point,-limits,limits)
        class Stop(Exception):pass
        def callback(x):
            nonlocal point
            point=x.copy();value,gradient=objective.evaluate(point)
            history.append(dict(iteration=len(history)+1,cycle=cycle,loss=value,
                                gradient_inf=float(np.abs(gradient).max()),seconds=time.perf_counter()-started))
            if len(history)%25==0:print(history[-1],flush=True)
            if time.perf_counter()-started>=seconds:raise Stop()
        try:
            result=minimize(objective.evaluate,point,jac=True,method='L-BFGS-B',bounds=Bounds(-limits,limits),
                            callback=callback,options=dict(maxiter=iterations_per_cycle,maxcor=20,maxls=40,ftol=0.,gtol=1e-9))
            point=result.x;solver_reason=str(result.message)
        except Stop:solver_reason='soft_time_limit'
        value,raw_gradient=objective.evaluate(point)
        with torch.no_grad():
            blocks=objective.unpack(torch.as_tensor(point,dtype=torch.float64,device=objective.device))
            bound=max(float(v.abs().max()) for v in blocks if v.numel())
        max_bound=max(max_bound,bound)
        assert bound<=1+1e-10
        state=objective.physical(point)
        fresh=Objective(target,state,total,penalty=penalty)
        fresh_value,gradient=fresh.evaluate(fresh.initial)
        replay=abs(fresh_value-value);max_replay=max(max_replay,replay)
        assert replay<=1e-8
        state=fresh.physical(fresh.initial)
        capture=1-fresh.last['residual'];grad=float(np.abs(gradient).max())
        progress=abs(history[-20]['loss']-fresh_value)/max(capture,1e-12) if len(history)>=20 else float('inf')
        converged=grad<=gradient_bar and progress<=progress_bar
        row=dict(cycle=cycle,before=before,after=fresh_value,raw_gradient_inf=float(np.abs(raw_gradient).max()),
                 fresh_gradient_inf=grad,relative_progress_20=progress if np.isfinite(progress) else None,
                 reencoding_objective_error=replay,maximum_raw_entry=bound,solver_termination=solver_reason,
                 local_converged=converged,inner_normal_residual=fresh.last['inner']['normal_residual'])
        cycles.append(row);print(dict(cycle_end=row),flush=True)
        if converged:reason='fresh_coordinate_convergence';break
        if time.perf_counter()-started>=seconds:reason='soft_time_limit';break
    sequence=[initial]+[r['loss'] for r in history]+[fresh_value]
    return state,dict(initial=initial,final=fresh_value,capture=capture,details=fresh.last,
                      fresh_gradient_inf=grad,relative_progress_20=progress if np.isfinite(progress) else None,
                      local_converged=converged,termination=reason,maximum_raw_entry=max_bound,
                      maximum_reencoding_objective_error=max_replay,
                      maximum_objective_increase=max([0.]+[b-a for a,b in zip(sequence,sequence[1:])]),
                      cycles=cycles,history=history,seconds=time.perf_counter()-started)
