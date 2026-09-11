"""Discrete shared-consumer reassignment followed by continuous refitting."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
from shared_reader_variable_projection_v2 import Objective
from shared_reader_variable_projection_v2_control import planted
from ll1_joint_parent_graph_v2 import build,execute,factors,price
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def fit(target,graph,total,initial=None):
    objective=Objective(target,graph,total,penalty=1e-6)
    if initial is None:initial=objective.initial
    started=time.perf_counter();accepted=initial.copy();iterations=0
    class Stop(Exception):pass
    def callback(point):
        nonlocal accepted,iterations
        accepted=point.copy();iterations+=1
        objective.evaluate(point)
        if time.perf_counter()-started>=60:raise Stop()
    initial_loss=objective.evaluate(initial)[0]
    try:
        result=minimize(objective.evaluate,initial,jac=True,method='L-BFGS-B',callback=callback,
                        options=dict(maxiter=1000,maxcor=20,maxls=40,ftol=0.,gtol=1e-9))
        accepted=result.x;termination=str(result.message)
    except Stop:termination='soft_time_limit'
    loss,gradient=objective.evaluate(accepted);fitted=objective.physical(accepted)
    tensor=dense(*cp(*factors(fitted)));target_tensor=dense(*target)
    error=float((tensor-target_tensor).norm()/target_tensor.norm())
    torch.manual_seed(2920);x=torch.randn(17,9)
    expected=torch.einsum('ode,nd,ne->no',tensor,x,x)
    replay=float((execute(fitted,x)-expected).norm()/expected.norm())
    return fitted,dict(initial_loss=initial_loss,loss=loss,gradient_inf=float(np.abs(gradient).max()),
                       relative_coefficient_error=error,termination=termination,iterations=iterations,
                       seconds=time.perf_counter()-started,graph_replay=replay,
                       inner_normal_residual=objective.last['inner']['normal_residual'])


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;target,graph=planted();total=float(dense(*target).square().sum())
    size=len(Objective(target,graph,total,penalty=1e-6).initial)
    fitted,baseline=fit(target,graph,total,np.random.default_rng(2916).normal(size=size))
    old=json.loads((root/'SHARED_READER_VARIABLE_PROJECTION_V2_PLANTED.json').read_text())
    prior=next(r for r in old['rows'] if r['seed']==2916)
    replay=abs(baseline['loss']-prior['final_loss'])
    print(json.dumps(dict(baseline=baseline,replay_loss_error=replay)),flush=True)
    parts=factors(fitted);rows=[];programs={}
    for consumers in ([0,2],[0,1],[1,2]):
        nodes=[dict(consumers=[0,1,2]),dict(consumers=consumers)]
        proposal=build(*parts,fitted['readers'],nodes)
        result,stats=fit(target,proposal,total)
        row=dict(consumers=consumers,**stats,price=price(result,parts),
                 error_reduction_fraction=1-stats['relative_coefficient_error']/baseline['relative_coefficient_error'])
        rows.append(row);programs[str(consumers)]=result
        print(json.dumps(row),flush=True)
    alternatives=rows[1:]
    result=dict(baseline=baseline,replay_loss_error=replay,rows=rows,
                pred_a=replay<=1e-10 and all(r['graph_replay']<=1e-8 and r['inner_normal_residual']<=1e-10 for r in rows),
                pred_b=any(r['error_reduction_fraction']>=.5 for r in alternatives),
                pred_c=any(r['relative_coefficient_error']<=.001 for r in alternatives),
                scope='Controlled consumer-topology moves on a known planted target, followed by joint reader/writer fitting and core elimination. A single selected failed start; no native/global recovery claim.')
    torch.save(programs,root/'SHARED_READER_CONSUMER_MOVES_V1.pt')
    (root/'SHARED_READER_CONSUMER_MOVES_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','rows')}),flush=True)


if __name__=='__main__':main()
