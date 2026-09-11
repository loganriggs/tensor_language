"""A feasibility/weakduality/gap<=1e-6; B planted rank1 rewrite recovered<=1e-5.
General signed planted example descriptive; convex optimality is not minimum rank.
"""
import json
from pathlib import Path
import torch
from quartic_gram_map_v1 import layout,coefficients,canonical
from quartic_gram_nuclear_v1 import solve


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(11531)
    out=Path(__file__).with_name('QUARTIC_GRAM_NUCLEAR_V1_CONTROL.json');assert not out.exists();reports=[]
    for rank in [2,4]:
        mapping=layout(rank);n=mapping['pairs'].shape[1]
        if rank==2:
            planted=torch.zeros(n,n);planted[1,1]=.5
        else:
            q=torch.linalg.qr(torch.randn(n,2)).Q;planted=(q*torch.tensor([1.,-.7]))@q.T
        c=coefficients(planted,mapping);base=canonical(c,mapping);x,r=solve(c,mapping)
        r.update(input_dimension=rank,canonical_rank=int(torch.linalg.matrix_rank(base)),canonical_nuclear_norm=float(torch.linalg.eigvalsh(base).abs().sum()),
            known_feasible_nuclear_norm=float(torch.linalg.eigvalsh(planted).abs().sum()),dual_exceeds_known_feasible=max(0.,r['dual_bound']-float(torch.linalg.eigvalsh(planted).abs().sum())))
        reports.append(r)
    result=dict(pred_a=all(r['coefficient_error']<=1e-10 and r['dual_exceeds_known_feasible']<=1e-8 and r['relative_gap']<=1e-6 and r['converged'] for r in reports),
        pred_b=reports[0]['effective_rank']==1 and reports[0]['truncated_coefficient_error']<=1e-5,reports=reports)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='reports'},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='history'} for r in reports],indent=2));assert result['pred_a'] and result['pred_b']


if __name__=='__main__':main()
