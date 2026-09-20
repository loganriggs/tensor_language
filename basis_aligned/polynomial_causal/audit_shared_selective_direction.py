"""Shared bounded intervention direction under per-input first-order constraints."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def solve(contexts):
    G=np.concatenate([np.array(c['gradient']) for c in contexts]);B=np.array([0.,0.,1.,1.,1.]);n=-G[:,0];modal=-G[:,1:];reference=n@B
    assert np.min(abs(reference))>1e-14
    signed=n*np.sign(reference)[:,None];scale=np.linalg.norm(signed,axis=1)
    rows=[]
    for nn,mm,ref,s in zip(signed,modal,abs(reference),scale):
        rows.append(np.r_[-nn/s,ref/s])
        for m in mm:
            rows.append(np.r_[(m-.1*nn)/s,0.]);rows.append(np.r_[(-m-.1*nn)/s,0.])
    matrix=np.array(rows);objective=np.array([0.,0.,0.,0.,0.,-1.]);bounds=[(-1,1)]*5+[(0,None)]
    r=linprog(objective,A_ub=matrix,b_ub=np.zeros(len(matrix)),bounds=bounds,method='highs');assert r.success,r.message
    stationarity=objective-matrix.T@r.ineqlin.marginals-r.lower.marginals-r.upper.marginals
    dual=-np.sum(r.lower.marginals[:5])+np.sum(r.upper.marginals[:5]);gap=abs(r.fun-dual)
    residual=max(float((matrix@r.x).max()),0.);assert residual<1e-8 and gap<1e-8 and abs(stationarity).max()<1e-8
    _,singular,Vt=np.linalg.svd(modal.reshape(-1,5),full_matrices=False)
    return dict(amplitudes=r.x[:5].tolist(),worst_linear_retention=float(r.x[-1]),constraint_residual=residual,primal_dual_gap=float(gap),dual_stationarity=float(abs(stationarity).max()),stacked_modal_rank=int(np.linalg.matrix_rank(modal.reshape(-1,5))),stacked_modal_singular_values=singular.tolist())
def main():
    old=json.loads((A/'five_source_modal_null_v1_result.json').read_text());train=[c for c in old['contexts'] if c['template'] in ['near_greeted','outside_called']]
    solutions={'common':solve(train)}
    for role in ['subject','attractor']:solutions[role]=solve([c for c in train if c['role']==role])
    stratified={}
    for role in ['subject','attractor']:
        for number in ['singular','plural']:
            selected=[]
            for c in train:
                if c['role']!=role:continue
                rows=json.loads((P/f"SEMANTIC_PORT_FRESH_{c['panel'].upper()}_ROWS.json").read_text());rows=[r for r in rows if r['template']==c['template']]
                ids=[i for i,r in enumerate(rows) if r['family'].endswith('|'+number)]
                selected.append(dict(gradient=[c['gradient'][i] for i in ids]))
            stratified[role+'_'+number]=solve(selected)
    (P/'SHARED_DIRECTION_NUMBER_STRATIFICATION.json').write_text(json.dumps(stratified,indent=2)+'\n')
    out=dict(solutions=solutions,scope='Frozen source-coordinate intervention directions from calibration derivatives only; first-order per-input constraints are stronger/different than finite group modal metrics. Zero optimum excludes positive uniform linear retention under these constraints only, not all nonlinear or shared circuits.')
    (P/'SHARED_SELECTIVE_DIRECTION_V1_RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
