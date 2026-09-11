"""Cached L-BFGS evaluations with explicit local convergence diagnostics."""
import time
import numpy as np
from scipy.optimize import minimize


def fit(objective,seconds=1200,max_iterations=10000):
    started=time.perf_counter();point=objective.initial.copy();history=[];cache={}
    def evaluate(x):
        if 'point' not in cache or not np.array_equal(x,cache['point']):
            value,gradient=objective.evaluate(x)
            cache.update(point=x.copy(),value=value,gradient=gradient.copy())
        return cache['value'],cache['gradient']
    initial,_=evaluate(point)
    class Stop(Exception):pass
    reason=''
    def accepted(x):
        nonlocal point,reason
        point=x.copy();value,gradient=evaluate(point)
        capture=1-objective.last['residual']
        row=dict(iteration=len(history)+1,seconds=time.perf_counter()-started,
                 loss=value,capture=capture,gradient_inf=float(np.max(np.abs(gradient))))
        history.append(row)
        progress=abs(history[-20]['loss']-value)/max(capture,1e-12) if len(history)>=20 else float('inf')
        row['relative_progress_20']=progress if np.isfinite(progress) else None
        if len(history)%25==0:print(row,flush=True)
        if row['gradient_inf']<=1e-7 and progress<=1e-6:
            reason='registered_local_convergence';raise Stop()
        if time.perf_counter()-started>=seconds:
            reason='soft_time_limit';raise Stop()
    try:
        result=minimize(evaluate,point,jac=True,method='L-BFGS-B',callback=accepted,
                        options=dict(maxiter=max_iterations,maxcor=20,maxls=40,ftol=0.,gtol=1e-9))
        point=result.x.copy();reason=str(result.message)
    except Stop:pass
    final,gradient=objective.evaluate(point)
    capture=1-objective.last['residual']
    progress=abs(history[-20]['loss']-final)/max(capture,1e-12) if len(history)>=20 else float('inf')
    grad=float(np.max(np.abs(gradient)))
    sequence=[initial]+[r['loss'] for r in history]+[final]
    return point,dict(initial=initial,final=final,details=objective.last,termination=reason,
                     gradient_inf=grad,relative_progress_20=progress if np.isfinite(progress) else None,
                     local_converged=bool(grad<=1e-7 and progress<=1e-6),
                     maximum_increase=max([0.]+[b-a for a,b in zip(sequence,sequence[1:])]),
                     seconds=time.perf_counter()-started,history=history,
                     scope='Registered first-order/progress criterion only; no global minimum or circuit identification guarantee.')
