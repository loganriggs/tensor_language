"""Eight registered starts for numeric recovery at a supplied shared DAG topology."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import torch
from shared_reader_variable_projection_v2 import Objective
from shared_reader_variable_projection_v2_control import planted
from ll1_joint_parent_graph_v2 import factors,execute
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


class TimeLimit(Exception):pass


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    target,graph=planted();tensor=dense(*target);total=float(tensor.square().sum())
    rows=[];root=Path(__file__).parent
    for seed in range(2910,2918):
        objective=Objective(target,graph,total,penalty=1e-6)
        rng=np.random.default_rng(seed);kind='near' if seed<2914 else 'independent'
        initial=objective.initial+.005*rng.normal(size=len(objective.initial)) if kind=='near' else rng.normal(size=len(objective.initial))
        start=time.perf_counter();accepted=initial.copy();history=[]
        def callback(point):
            nonlocal accepted
            accepted=point.copy();value,gradient=objective.evaluate(point)
            history.append(dict(iteration=len(history)+1,loss=value,gradient_inf=float(np.abs(gradient).max())))
            if time.perf_counter()-start>=60:raise TimeLimit
        initial_loss=objective.evaluate(initial)[0]
        try:
            result=minimize(objective.evaluate,initial,jac=True,method='L-BFGS-B',callback=callback,
                            options=dict(maxiter=1000,maxcor=20,maxls=40,ftol=0.,gtol=1e-9))
            accepted=result.x;termination=str(result.message)
        except TimeLimit:termination='soft_time_limit'
        final,gradient=objective.evaluate(accepted)
        fitted=objective.physical(accepted);actual=dense(*cp(*factors(fitted)))
        error=float((actual-tensor).norm()/tensor.norm())
        torch.manual_seed(2918);x=torch.randn(17,9)
        reference=torch.einsum('ode,nd,ne->no',actual,x,x)
        replay=float((execute(fitted,x)-reference).norm()/reference.norm())
        row=dict(seed=seed,kind=kind,initial_loss=initial_loss,final_loss=final,
                 relative_coefficient_error=error,recovered=error<=.001,
                 gradient_inf=float(np.abs(gradient).max()),termination=termination,
                 iterations=len(history),seconds=time.perf_counter()-start,
                 graph_replay=replay,inner_normal_residual=objective.last['inner']['normal_residual'],
                 shared_reader_abs_cosines=(fitted['readers']*graph['readers']).sum(1).abs().tolist(),
                 history=history)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='history'}),flush=True)
        # Preserve each completed row while subsequent starts run.
        (root/'SHARED_READER_VARIABLE_PROJECTION_V2_PLANTED_PROGRESS.json').write_text(json.dumps(rows,indent=2)+'\n')
    result=dict(rows=rows,pred_a=all(r['graph_replay']<=1e-8 and r['inner_normal_residual']<=1e-10 for r in rows),
                pred_b=sum(r['recovered'] for r in rows if r['kind']=='near')>=3,
                pred_c=sum(r['recovered'] for r in rows if r['kind']=='independent')>=2,
                scope='Numeric weight-tensor recovery with supplied shared topology. Near/independent initializations separated; no topology discovery, native model recovery or behavioral circuit claim.')
    (root/'SHARED_READER_VARIABLE_PROJECTION_V2_PLANTED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
