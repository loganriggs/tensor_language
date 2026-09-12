"""Single-node independent replacements after a planted stationary miss.

A numerical replay/constraints/descent; B any recovery <=1e-4;
C >=2/4 recoveries replacing least conditionally contributing node.
"""
import json
from pathlib import Path
import torch
from coupled_quartic_writer_v1 import gram
from quartic_manifold_lbfgs_v1 import fit
from planted_quartic_objective_v1 import objective

P=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
saved=torch.load(P/'QUARTIC_LBFGS_STATIONARY_MISS.pt',weights_only=True)
b,n=saved['b'],saved['n']
evaluate,energy=objective(saved['trueb'],saved['truen'],torch.tensor([[1.,.3],[-.2,.9]]))
value,mixing=evaluate(b,n,energy)
previous=json.loads((P/'QUARTIC_LBFGS_INDEPENDENT_V1_CONTROL.json').read_text())['reports'][-1]
replay=abs(max(0.,value)**.5-previous['relative_coefficient_error'])
k=gram(b,n);conditional=mixing.square().sum(-1)/torch.linalg.inv(k).diagonal()
weak=int(conditional.argmin());reports=[]
for node in range(len(b)):
    for seed in range(91831,91835):
        torch.manual_seed(seed);bn,nn=b.clone(),n.clone()
        bn[node]=torch.linalg.qr(torch.randn_like(bn[node]),mode='reduced')[0]
        weights=torch.randn_like(nn[node]);nn[node]=weights/weights.norm()
        bn,nn,mix,h,reason=fit(bn,nn,evaluate,energy,max_steps=1000,max_seconds=30)
        constraint=max(float((bn.transpose(-1,-2)@bn-torch.eye(2)).abs().max()),float((nn.norm(dim=-1)-1).abs().max()))
        monotonic=all(z['objective']<=h[i]['objective']+1e-10 for i,z in enumerate(h[1:]))
        r=dict(node=node,seed=seed,error=max(0.,h[-1]['objective'])**.5,gradient=h[-1]['projected_gradient_norm'],iterations=len(h),seconds=h[-1]['seconds'],termination=reason,constraint_error=constraint,monotonic=monotonic)
        reports.append(r);print(json.dumps(r),flush=True)
result=dict(pred_a=replay<=1e-10 and all(r['constraint_error']<=1e-10 and r['monotonic'] for r in reports),
            pred_b=any(r['error']<=1e-4 for r in reports),
            pred_c=sum(r['error']<=1e-4 for r in reports if r['node']==weak)>=2,
            weak_node=weak,conditional_contribution=conditional.tolist(),replay_error=replay,reports=reports,
            scope='Local replacement heuristic on one synthetic failed endpoint, no native or general recovery guarantee.')
out=P/'QUARTIC_NODE_REPLACEMENT_V1_CONTROL.json';assert not out.exists()
out.write_text(json.dumps(result,indent=2)+'\n');assert result['pred_a']
