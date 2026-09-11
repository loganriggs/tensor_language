"""Distinguish optimizer restart from same-function coordinate re-encoding."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
from shared_reader_variable_projection_v2 import Objective
from shared_reader_variable_projection_v2_control import planted
from ll1_joint_parent_graph_v2 import build,factors
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def optimize(objective,initial):
    start=time.perf_counter();point=initial.copy();steps=0
    class Stop(Exception):pass
    def callback(x):
        nonlocal point,steps
        point=x.copy();steps+=1;objective.evaluate(x)
        if time.perf_counter()-start>=60:raise Stop()
    try:
        result=minimize(objective.evaluate,initial,jac=True,method='L-BFGS-B',callback=callback,
                        options=dict(maxiter=1000,maxcor=20,maxls=40,ftol=0.,gtol=1e-9))
        point=result.x;reason=str(result.message)
    except Stop:reason='soft_time_limit'
    loss,gradient=objective.evaluate(point)
    return point,dict(loss=loss,gradient_inf=float(np.abs(gradient).max()),steps=steps,
                      seconds=time.perf_counter()-start,termination=reason)


def reencode(graph):
    nodes=[dict(consumers=[g for g,group in enumerate(graph['groups']) if i in group['parent_ids'].tolist()])
           for i in range(len(graph['readers']))]
    return build(*factors(graph),graph['readers'],nodes)


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;target,truth=planted();target_tensor=dense(*target);total=float(target_tensor.square().sum())
    old=json.loads((root/'SHARED_READER_VARIABLE_PROJECTION_V2_PLANTED.json').read_text())
    rows=[];raw_restart=None
    for seed in range(2914,2918):
        graph=truth;objective=Objective(target,graph,total,penalty=1e-6)
        point=np.random.default_rng(seed).normal(size=len(objective.initial));cycles=[]
        for cycle in range(4):
            point,stats=optimize(objective,point)
            physical=objective.physical(point);tensor=dense(*cp(*factors(physical)))
            error=float((tensor-target_tensor).norm()/target_tensor.norm())
            baseline=next(r for r in old['rows'] if r['seed']==seed)
            baseline_replay=abs(stats['loss']-baseline['final_loss']) if cycle==0 else None
            raw_parts=objective.unpack(torch.tensor(point))
            scales=[dict(min_row_norm=float(p.norm(dim=1).min()),max_row_norm=float(p.norm(dim=1).max())) for p in raw_parts]
            if seed==2916 and cycle==0:
                raw_point,raw_stats=optimize(objective,point)
                raw_tensor=dense(*cp(*factors(objective.physical(raw_point))))
                raw_restart=dict(**raw_stats,relative_coefficient_error=float((raw_tensor-target_tensor).norm()/target_tensor.norm()))
                # Continue the re-encoding comparison from the original endpoint.
                objective.evaluate(point)
            encoded=reencode(physical);encoded_tensor=dense(*cp(*factors(encoded)))
            function_replay=float((encoded_tensor-tensor).norm()/tensor.norm())
            next_objective=Objective(target,encoded,total,penalty=1e-6)
            new_loss,new_gradient=next_objective.evaluate(next_objective.initial)
            entry=dict(cycle=cycle,**stats,relative_coefficient_error=error,
                       baseline_objective_replay=baseline_replay,raw_row_scales=scales,
                       reencoding_function_error=function_replay,reencoding_objective_error=abs(new_loss-stats['loss']),
                       reencoded_gradient_inf=float(np.abs(new_gradient).max()),
                       inner_normal_residual=objective.last['inner']['normal_residual'])
            cycles.append(entry)
            print(json.dumps(dict(seed=seed,**{k:v for k,v in entry.items() if k!='raw_row_scales'})),flush=True)
            if error<=.001:break
            objective=next_objective;point=objective.initial
        row=dict(seed=seed,cycles=cycles,recovered=cycles[-1]['relative_coefficient_error']<=.001)
        rows.append(row)
    seed2916=next(r for r in rows if r['seed']==2916)
    result=dict(rows=rows,raw_restart=raw_restart,
                pred_a=all(c['reencoding_function_error']<=1e-8 and c['reencoding_objective_error']<=1e-8 for r in rows for c in r['cycles']),
                pred_b=raw_restart['relative_coefficient_error']>.001 and seed2916['recovered'],
                pred_c=sum(r['recovered'] for r in rows)>=2,
                scope='Same topology/function re-encoded between L-BFGS solves. Finite tolerance gradients depend on numerical coordinates; original independent-start failure remains recorded. Not a global convergence theorem.')
    (root/'SHARED_READER_GAUGE_RESTART_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
