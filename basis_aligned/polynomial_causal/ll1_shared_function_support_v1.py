"""Sparse conditional OLS support and shared-square/product accounting."""
import json
from pathlib import Path
import torch
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner
from ll1_group_matching_v1 import parent_cp
from ll1_shared_parent_v1 import shared_parent
from ll1_distributed_function_v1 import component_cross


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    native=tuple(x.double() for x in torch.load('/dev/shm/bilin18_matched_shared_groups_v1_ll1_native.pt',weights_only=True,map_location='cpu')['parts'])
    spectral=tuple(x.double() for x in torch.load('/dev/shm/bilin18_matched_shared_groups_v1_ll1_spectral.pt',weights_only=True,map_location='cpu')['parts'])
    pair=json.loads((root/'LL1_GROUP_MATCHING_V1_AUDIT.json').read_text())['native_pair']
    move=shared_parent(tuple(x[pair] for x in native));target=parent_cp(move)
    total=inner(target,target)
    square=tuple(v[:1] if i<2 else v[:,:1] for i,v in enumerate(target))
    mixed=tuple(v[1:] if i<2 else v[:,1:] for i,v in enumerate(target))
    es=inner(square,square);em=inner(mixed,mixed);cross=2*inner(square,mixed)
    bank=cp(*spectral);gram=component_cross(bank,bank);norms=gram.diag().sqrt()
    g=gram/norms[:,None]/norms[None,:]
    h=component_cross(bank,target).sum(1)/norms/total.sqrt()
    support=[];rows=[];beta=torch.zeros(0)
    for step in range(1,33):
        if support:
            sub=g[support][:,support];crossg=g[:,support]
            inverse_cross=torch.linalg.solve(sub,crossg.T)
            remaining=1-(crossg*inverse_cross.T).sum(1)
            residual=h-crossg@beta
        else:remaining=g.diag().clone();residual=h.clone()
        gains=residual.square()/remaining.clamp_min(1e-12)
        gains[remaining<=1e-10]=-torch.inf
        if support:gains[support]=-torch.inf
        support.append(int(gains.argmax()))
        sub=g[support][:,support];beta=torch.linalg.solve(sub,h[support])
        if step in (1,2,4,8,16,32):
            capture=float(beta@h[support]);weights=beta*total.sqrt()/norms[support]
            fitted=(bank[0][support],bank[1][support],bank[2][:,support]*weights)
            actual=float((total-2*inner(fitted,target)+inner(fitted,fitted))/total)
            rows.append(dict(components=step,function_cosine=max(0.,capture)**.5,squared_relative_error=actual,
                             exact_projection_replay=abs(actual-(1-capture)),support=list(support),
                             groups=sorted(set(i//16 for i in support)),weights=weights.tolist()))
    selected=next((r for r in rows if r['function_cosine']>=.99),rows[-1])
    identity=float(abs(es+em+cross-total)/total)
    result=dict(pred_a=max([identity]+[r['exact_projection_replay'] for r in rows])<=1e-10,
                pred_b=selected['components']<=16 and selected['function_cosine']>=.99,
                pred_c=len(selected['groups'])>=2,square_energy_fraction=float(es/total),
                mixed_product_energy_fraction=float(em/total),square_mixed_cross_fraction=float(cross/total),
                energy_identity_error=identity,selected_components=selected['components'],selected_groups=selected['groups'],rows=rows,
                scope='Greedy exact conditional OLS, not minimum-support proof. Target is the previously selected shared-parent function; weight-only cross-start screen, no behavioral independence or semantic claim.')
    (root/'LL1_SHARED_FUNCTION_SUPPORT_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if k not in ('support','weights')} for r in rows]},indent=2))


if __name__=='__main__':main()
