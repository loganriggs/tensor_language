"""Product-domain error witnesses; local lower bounds, never global optima."""
import json
import signal
import time
from datetime import datetime, timezone
import torch
from retained_objective_context_v1 import Contexts, P


def unit(x):
    return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-30)


@torch.no_grad()
def main():
    out=P/'BILINEAR_INJECTIVE_SCREEN_V1_RESULT.json';assert not out.exists()
    torch.set_num_threads(2);signal.alarm(120);start=time.monotonic()
    model=Contexts();rows=[]
    for name in ('SPARSE_INTERACTION_EXECUTOR_V1','MINIMAX_ROW_SUPPORT_V1'):
        e=model.decode(name)-model.tensor
        generator=torch.Generator().manual_seed(170231000)
        u=unit(torch.randn(32,12,generator=generator,dtype=torch.float64))
        z=unit(torch.randn(32,1152,generator=generator,dtype=torch.float64))
        a=unit(torch.randn(32,128,generator=generator,dtype=torch.float64))
        for _ in range(80):
            u=unit(torch.einsum('oih,ni,nh->no',e,z,a))
            z=unit(torch.einsum('oih,no,nh->ni',e,u,a))
            a=unit(torch.einsum('oih,no,ni->nh',e,u,z))
        outputs=torch.einsum('oih,ni,nh->no',e,z,a)
        objective=(outputs*u).sum(-1).abs()
        direct=torch.einsum('oih,no,ni,nh->n',e,u,z,a).abs()
        replay=float((objective-direct).norm()/objective.norm())
        next_u=unit(outputs)
        next_z=unit(torch.einsum('oih,no,nh->ni',e,u,a))
        next_a=unit(torch.einsum('oih,no,ni->nh',e,u,z))
        stationarity=torch.stack(((next_u-u).norm(dim=1),(next_z-z).norm(dim=1),(next_a-a).norm(dim=1))).max(0).values
        flat=e.flatten(1)
        upper=float(torch.linalg.eigvalsh(flat@flat.T)[-1].clamp_min(0).sqrt())
        rows.append(dict(name=name,witness_values=objective.tolist(),best_lower_bound=float(objective.max()),
                         output_unfolding_upper=upper,best_stationarity=float(stationarity[objective.argmax()]),
                         all_stationarity=stationarity.tolist(),replay=replay))
        print(json.dumps(rows[-1]),flush=True)
    result=dict(utc=datetime.now(timezone.utc).isoformat(),rows=rows,
                attained_bound_ratio=rows[1]['best_lower_bound']/rows[0]['best_lower_bound'],
                pred_a=all(r['replay']<=1e-10 and r['best_lower_bound']<=r['output_unfolding_upper']*(1+1e-10) for r in rows),
                pred_b=rows[1]['best_lower_bound']<=.9*rows[0]['best_lower_bound'],
                seconds=time.monotonic()-start,
                scope='32local alternating witnesses per frozen error tensor; lower/upper bounds only, no global optimality, fitting, native transfer or adoption.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    signal.alarm(0);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
