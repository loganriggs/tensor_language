"""Match LL1 groups as signed tensor functions, then track shared-parent reuse."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner
from ll1_shared_parent_v1 import shared_parent


def group_inner(first,second):
    a,s,c=first;b,t,d=second
    overlaps=(a.flatten(0,1)@b.flatten(0,1).T).reshape(len(a),a.shape[1],len(b),b.shape[1])
    q=torch.einsum('irjs,ir,js->ij',overlaps.square(),s,t)
    return q*(c@d.T)


def parent_cp(move):
    u=move['parent'];groups=move['groups']
    a=[u];b=[u];w=[sum(g['alpha']*g['writer'] for g in groups)]
    for g in groups:
        a.append(u);b.append(g['private']@g['beta']);w.append(2*g['writer'])
    return torch.stack(a),torch.stack(b),torch.stack(w,1)


def cosine(a,b):
    return float(inner(a,b)/(inner(a,a)*inner(b,b)).sqrt())


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    parts={label:tuple(x.double() for x in torch.load(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt',weights_only=True,map_location='cpu')['parts']) for label in ('spectral','native')}
    left,right=parts['spectral'],parts['native']
    cross=group_inner(left,right);ll=group_inner(left,left);rr=group_inner(right,right)
    sims=cross/(ll.diag()[:,None]*rr.diag()[None,:]).sqrt()
    si,ni=linear_sum_assignment(-sims.numpy())
    mapping={int(n):int(s) for s,n in zip(si,ni)}
    original=json.loads((root/'LL1_SHARED_PARENT_V1_AUDIT.json').read_text())
    native_pair=next(r['pair'] for r in original['rows'] if r['label']=='native')
    spectral_pair=[mapping[i] for i in native_pair]
    pair_parts=[tuple(x[ids] for x in part) for part,ids in ((left,spectral_pair),(right,native_pair))]
    moves=[shared_parent(part) for part in pair_parts]
    matches=[float(sims[s,n]) for s,n in zip(spectral_pair,native_pair)]
    parent_cos=float(abs(moves[0]['parent']@moves[1]['parent']))
    output_cos=cosine(*(parent_cp(m) for m in moves))
    independent=float(inner(cp(*(x[[spectral_pair[0]]] for x in left)),cp(*(x[[native_pair[0]]] for x in right))))
    replay=abs(independent-float(cross[spectral_pair[0],native_pair[0]]))/max(1.,abs(independent))
    selferror=float(((ll/(ll.diag()[:,None]*ll.diag()[None,:]).sqrt()).diag()-1).abs().max())
    matched=sims[si,ni]
    result=dict(pred_a=max(replay,selferror)<1e-9,pred_b=min(matches)>=.8,pred_c=parent_cos>=.9 and output_cos>=.8,
                spectral_pair=spectral_pair,native_pair=native_pair,matched_pair_group_cosines=matches,
                matched_group_cosine_quantiles=torch.quantile(matched,torch.tensor([0.,.25,.5,.75,1.])).tolist(),
                groups_ge_08=int((matched>=.8).sum()),groups_ge_09=int((matched>=.9).sum()),
                shared_parent_abs_cosine=parent_cos,shared_output_function_cosine=output_cos,
                matched_two_group_sum_cosine=cosine(*(cp(*p) for p in pair_parts)),
                within_pair_principal_cosines=[m['cosine'] for m in moves],
                independent_inner_replay=replay,self_cosine_error=selferror,
                matching=[dict(spectral=int(s),native=int(n),cosine=float(sims[s,n])) for s,n in zip(si,ni)],
                scope='Weight-only post-fit one-to-one group alignment and tracked pair screen, not held-out behavioral identification; pilots unconverged.')
    (root/'LL1_GROUP_MATCHING_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='matching'},indent=2))


if __name__=='__main__':main()
