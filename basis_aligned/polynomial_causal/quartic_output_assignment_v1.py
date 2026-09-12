"""Red-team exclusive output assignment by exact single-flip refits.

Fixed V3 readers and diagnosed output angle; no target/model/data refitting.
Prediction: a greedy assignment repair reduces fitted-program error below .1.
Failure means only this local fixed-angle search failed, not necessary reuse.
"""
import json
import torch
from quartic_output_reuse_v1 import P, rotation
from coupled_quartic_writer_v1 import gram, solve

torch.set_num_threads(2)
prior=json.loads((P/'QUARTIC_OUTPUT_REUSE_V1_RESULT.json').read_text())['reports'][-1]
program=torch.load(P/'COUPLED_QUARTIC_NONLINEAR_V3_PROGRAM.pt',weights_only=True)
k=gram(program['input_readers'],program['inner_weights'])
a=program['mixing']@rotation(prior['angle']); cross=k@a
energy=float((a*cross).sum())
def fit(assignment):
    b=torch.zeros_like(a)
    for branch in (0,1):
        ids=torch.where(assignment==branch)[0]
        if len(ids):
            values,_=solve(k[ids][:,ids],cross[ids,branch:branch+1])
            b[ids,branch]=values[:,0]
    d=b-a
    return float((d*(k@d)).sum())/energy
assignment=a.abs().argmax(-1); initial=fit(assignment); current=initial; history=[]
for iteration in range(65):
    proposals=[]
    for node in range(len(a)):
        candidate=assignment.clone();candidate[node]=1-candidate[node]
        proposals.append(fit(candidate))
    best=min(range(len(a)),key=proposals.__getitem__)
    if proposals[best]>=current-1e-12 or iteration==64:
        break
    assignment[best]=1-assignment[best];current=proposals[best]
    history.append(dict(node=best,relative_error=current**.5))
result=dict(initial_error=initial**.5,final_error=current**.5,history=history,
            initial_replay_error=abs(initial**.5-prior['exclusive_refit_relative_error']),
            single_flip_stationary=min(proposals)>=current-1e-12,
            best_remaining_improvement=current-min(proposals),
            pred_repair=current**.5<.1,
            scope='Fixed output angle, greedy exact single-flip assignment; no global guarantee or semantic claim.')
assert result['initial_replay_error']<1e-10
out=P/'QUARTIC_OUTPUT_ASSIGNMENT_V1_RESULT.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
