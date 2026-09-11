"""Exact linear-span red-team of cross-start LL1 group mismatch."""
import json
from pathlib import Path
import torch
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner
from ll1_group_matching_v1 import group_inner, parent_cp
from ll1_shared_parent_v1 import shared_parent


def component_cross(a,b):
    x,y,w=a;u,v,c=b
    return ((x@u.T)*(y@v.T)+(x@v.T)*(y@u.T))*.5*(w.T@c)


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    parts={label:tuple(x.double() for x in torch.load(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt',weights_only=True,map_location='cpu')['parts']) for label in ('spectral','native')}
    pair=json.loads((root/'LL1_GROUP_MATCHING_V1_AUDIT.json').read_text())['native_pair']
    native=parts['native'];spectral=parts['spectral'];bank=cp(*spectral)
    targets={f'group_{i}':cp(*(x[[i]] for x in native)) for i in pair}
    targets['pair_sum']=cp(*(x[pair] for x in native))
    targets['shared_parent']=parent_cp(shared_parent(tuple(x[pair] for x in native)))
    component_gram=component_cross(bank,bank);rows=[]
    for mode in ('whole_groups','square_components'):
        gram=group_inner(spectral,spectral) if mode=='whole_groups' else component_gram
        norms=gram.diag().sqrt();unit_gram=gram/norms[:,None]/norms[None,:]
        values,vectors=torch.linalg.eigh(unit_gram)
        keep=values>values[-1]*1e-10
        for name,target in targets.items():
            target_norm=inner(target,target).sqrt()
            h=component_cross(bank,target).sum(1)
            if mode=='whole_groups':h=h.reshape(64,16).sum(1)
            h=h/norms/target_norm
            beta=vectors[:,keep]@((vectors[:,keep].T@h)/values[keep])
            normal=float((unit_gram@beta-h).norm()/h.norm())
            capture=float(h@beta);scale=beta*target_norm/norms
            if mode=='whole_groups':scale=scale.repeat_interleave(16)
            fitted=(bank[0],bank[1],bank[2]*scale)
            independent_error=float((inner(target,target)-2*inner(fitted,target)+inner(fitted,fitted))/target_norm.square())
            replay=abs(independent_error-(1-capture))
            row=dict(mode=mode,target=name,numerical_rank=int(keep.sum()),
                     squared_relative_residual=independent_error,maximum_function_cosine=max(0.,capture)**.5,
                     normalized_coefficient_energy=float(beta.square().sum()),normal_residual=normal,
                     independent_cp_replay=replay,pred_a=max(normal,replay)<1e-8)
            rows.append(row)
            print(json.dumps(row),flush=True)
    parent={r['mode']:r for r in rows if r['target']=='shared_parent'}
    result=dict(pred_a=all(r['pred_a'] for r in rows),pred_b=parent['whole_groups']['maximum_function_cosine']>=.9,
                pred_c=parent['square_components']['maximum_function_cosine']>=.9,rows=rows,
                scope='Best normalized coefficient-space linear projection onto frozen other-start group or fixed-writer component spans. Not a nonlinear basis search, behavioral preservation test, or proof of absent native structure.')
    (root/'LL1_DISTRIBUTED_FUNCTION_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
