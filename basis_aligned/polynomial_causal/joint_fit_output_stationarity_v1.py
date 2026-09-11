"""Exact conditional writer check on completed core-eliminated spectral fits.

A receipt/normal/minimum identities <=1e-9.
B exact output-only objective gain >=1e-4 in both families.
C whole-function relative change >=.05 in both families.
"""
import hashlib
import json
from pathlib import Path
import torch
from ll1_joint_parent_graph_v2 import factors
from equilibrated_ll1_projected_v1 import output_solve


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;output=root/'JOINT_FIT_OUTPUT_STATIONARITY_V1.json'
    assert not output.exists()
    binding=json.loads((root/'SHARED_NODE_PARENT1_FINEWEB_V1_BINDING.json').read_text())['files']
    checkpoint=next(p for p in binding if p.endswith('/pytorch_model.bin'))
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    l,r,d=[weights[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    rows=[];total=99245061353.47293;target=None
    for family in ('ORIGINAL','GRAPH'):
        path=root/f'SHARED_READER_JOINT_FIT_V1_SPECTRAL_{family}.pt'
        graph=torch.load(path,weights_only=True,map_location='cpu')
        receipt=json.loads(path.with_suffix('.json').read_text())
        if target is None:target=(l,r,graph['output_whitener']@d)
        a,s,c=factors(graph)
        new,gram,rhs,diagnostics=output_solve(target,a,s,.01)
        def objective(w):
            return 1+(-2*(rhs.T*w).sum()+(gram*(w@w.T)).sum()+.01*(gram.diag()*w.square().sum(1)).sum())/total
        old_loss,new_loss=objective(c),objective(new)
        identity=1-(rhs.T*new).sum()/total
        difference=new-c
        change=((gram*(difference@difference.T)).sum()/(gram*(c@c.T)).sum()).clamp_min(0).sqrt()
        rows.append(dict(family=family,original_objective=float(old_loss),conditional_objective=float(new_loss),
            objective_gain=float(old_loss-new_loss),relative_function_change=float(change),
            receipt_replay=abs(float(old_loss)-receipt['optimization']['final']),
            minimum_identity_error=float(abs(new_loss-identity)),**diagnostics,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    result=dict(pred_a=all(max(r['receipt_replay'],r['minimum_identity_error'],r['output_normal_residual'])<=1e-9 for r in rows),
        pred_b=all(r['objective_gain']>=1e-4 for r in rows),pred_c=all(r['relative_function_change']>=.05 for r in rows),
        rows=rows,scope='Exact conditional output-only diagnostic; inputs held fixed. '
              'No joint convergence, global recovery, or circuit claim; no fitted artifact adopted.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
