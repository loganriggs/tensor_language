"""Same-function independent planted starts under bounded coordinate fitting."""
import json
from pathlib import Path
import numpy as np
import torch
from bounded_shared_reader_fit_v1 import fit
from shared_reader_variable_projection_v2 import Objective
from shared_reader_variable_projection_v2_control import planted
from ll1_joint_parent_graph_v2 import factors,execute
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;target,truth=planted();tensor=dense(*target);total=float(tensor.square().sum())
    rows=[]
    for seed in range(2914,2918):
        old=Objective(target,truth,total,penalty=1e-6)
        random=np.random.default_rng(seed).normal(size=len(old.initial))
        initial_loss=old.evaluate(random)[0];graph=old.physical(random)
        fitted,stats=fit(target,graph,total,penalty=1e-6,seconds=60,max_cycles=20,iterations_per_cycle=200)
        actual=dense(*cp(*factors(fitted)));error=float((actual-tensor).norm()/tensor.norm())
        torch.manual_seed(3011);x=torch.randn(13,9);reference=torch.einsum('ode,nd,ne->no',actual,x,x)
        replay=float((execute(fitted,x)-reference).norm()/reference.norm())
        row=dict(seed=seed,relative_coefficient_error=error,recovered=error<=.001,
                 initial_function_objective_replay=abs(initial_loss-stats['initial']),
                 executor_replay=replay,optimization=stats)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='optimization'}),flush=True)
    result=dict(rows=rows,
                pred_a=all(max(r['initial_function_objective_replay'],r['optimization']['maximum_reencoding_objective_error'],r['executor_replay'])<=1e-8 and r['optimization']['maximum_raw_entry']<=1+1e-10 for r in rows),
                pred_b=sum(r['recovered'] for r in rows)>=2,
                pred_c=all('fresh_gradient_inf' in r['optimization'] for r in rows),
                scope='Bounded raw-entry coordinates preserve normalized-reader representability. Same supplied topology and initial functions as independent-start controls; no global recovery guarantee.')
    (root/'BOUNDED_SHARED_READER_FIT_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
