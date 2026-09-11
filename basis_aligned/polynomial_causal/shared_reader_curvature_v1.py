"""Finite-difference Hessian and downhill escape at fresh-coordinate misses."""
import json
from pathlib import Path
import numpy as np
import torch
from bounded_shared_reader_fit_v1 import fit
from shared_reader_variable_projection_v2 import Objective
from shared_reader_variable_projection_v2_control import planted
from ll1_joint_parent_graph_v2 import factors
from symmetric_ll1_objective_v1 import cp
from chunked_bilinear_coefficient_v1 import dense


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;target,truth=planted();tensor=dense(*target);total=float(tensor.square().sum())
    prior=json.loads((root/'BOUNDED_SHARED_READER_FIT_V1_CONTROL.json').read_text())
    rows=[]
    for seed in (2915,2916):
        objective=Objective(target,truth,total,penalty=1e-6)
        initial=np.random.default_rng(seed).normal(size=len(objective.initial))
        objective.evaluate(initial);graph=objective.physical(initial)
        graph,stats=fit(target,graph,total,penalty=1e-6,seconds=60,max_cycles=20)
        old=next(r for r in prior['rows'] if r['seed']==seed)
        baseline_replay=abs(stats['final']-old['optimization']['final'])
        objective=Objective(target,graph,total,penalty=1e-6)
        point=objective.initial;value,gradient=objective.evaluate(point)
        h=1e-4;columns=[]
        for i in range(len(point)):
            direction=np.zeros_like(point);direction[i]=1.
            columns.append((objective.evaluate(point+h*direction)[1]-objective.evaluate(point-h*direction)[1])/(2*h))
        matrix=np.stack(columns,axis=1)
        asymmetry=float(np.linalg.norm(matrix-matrix.T)/np.linalg.norm(matrix))
        eigenvalues,eigenvectors=np.linalg.eigh((matrix+matrix.T)/2)
        direction=eigenvectors[:,0];step=1e-3
        curvature=(objective.evaluate(point+step*direction)[0]+objective.evaluate(point-step*direction)[0]-2*value)/step**2
        agreement=abs(curvature-eigenvalues[0])/max(abs(curvature),abs(eigenvalues[0]),1e-8)
        credible=eigenvalues[0]<-1e-4 and agreement<=.1
        recovery=None;trials=[]
        if credible:
            for distance in (.01,.03,.1):
                for sign in (-1,1):
                    shifted=point+sign*distance*direction
                    trials.append((objective.evaluate(shifted)[0],sign*distance,shifted))
            chosen=min(trials,key=lambda item:item[0]);objective.evaluate(chosen[2])
            proposal=objective.physical(chosen[2])
            fitted,new_stats=fit(target,proposal,total,penalty=1e-6,seconds=60,max_cycles=20)
            actual=dense(*cp(*factors(fitted)));error=float((actual-tensor).norm()/tensor.norm())
            recovery=dict(chosen_step=chosen[1],starting_objective=chosen[0],relative_coefficient_error=error,
                          fresh_gradient_inf=new_stats['fresh_gradient_inf'],local_converged=new_stats['local_converged'],
                          cycles=len(new_stats['cycles']))
        row=dict(seed=seed,baseline_objective_replay=baseline_replay,baseline_relative_error=old['relative_coefficient_error'],
                 fresh_gradient_inf=float(np.abs(gradient).max()),hessian_relative_asymmetry=asymmetry,
                 smallest_eigenvalue=float(eigenvalues[0]),objective_curvature=float(curvature),
                 curvature_relative_agreement=float(agreement),credible_negative_curvature=bool(credible),
                 eigenvalues=eigenvalues.tolist(),recovery=recovery,
                 pred_a=baseline_replay<=1e-8 and asymmetry<=1e-3)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='eigenvalues'}),flush=True)
    result=dict(rows=rows,pred_a=all(r['pred_a'] for r in rows),
                pred_b=any(r['credible_negative_curvature'] for r in rows),
                pred_c=any(r['recovery'] is not None and r['recovery']['relative_coefficient_error']<=.001 for r in rows),
                scope='Finite curvature check on two selected planted misses, followed by a registered negative-curvature move only when independently corroborated. No global/native convergence theorem.')
    (root/'SHARED_READER_CURVATURE_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
