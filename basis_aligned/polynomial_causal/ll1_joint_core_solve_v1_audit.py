"""Matched converged conditional core fits: original versus compatible DAG."""
import json
from pathlib import Path
import torch
from ll1_joint_core_solve_v1 import coordinates,System,solve,install,control
from ll1_joint_parent_graph_v2 import build,execute,factors,price
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent;check=control()
    print(json.dumps(dict(control=check)),flush=True)
    graphs=torch.load(root/'LL1_COMPATIBLE_PARENTS_V1.pt',weights_only=True)
    checkpoint=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
    sd=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    left,right,down=(sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down'))
    total=99245061353.47293;rows=[];artifacts={}
    for label in ('spectral','native'):
        saved=torch.load(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt',weights_only=True)
        original=tuple(v.double() for v in saved['parts'])
        target=(left,right,saved['output_whitener'].double()@down)
        arms={};artifacts[label]={}
        for name,graph in [('original',build(*original,torch.zeros(0,original[0].shape[-1]),[])),('graph',graphs[label])]:
            bases,writers,initial=coordinates(graph)
            orth=float((bases.transpose(1,2)@bases-torch.eye(bases.shape[-1])).norm()/len(bases)**.5)
            assert orth<=1e-10,orth
            system=System(bases,writers,target)
            def objective(core):return float(1+((core*system.apply(core)).sum()-2*(core*system.rhs).sum())/total)
            before=objective(initial)
            fitted,stats=solve(system,initial)
            after=objective(fitted)
            updated=install(graph,fitted,writers)
            updated.update(output_whitener=saved['output_whitener'].double(),bias=sd['transformer.h.17.mlp.Down_bias'].clone())
            candidate=cp(*factors(updated))
            capture=float((2*inner(target,candidate)-inner(candidate,candidate))/total)
            energy=float(fitted.square().sum()/total)
            replay_objective=abs(after-(1-capture+.01*energy))
            torch.manual_seed(2802);x=torch.randn(13,bases.shape[1])
            reference=((x@candidate[0].T)*(x@candidate[1].T))@candidate[2].T
            replay=float((execute(updated,x)-reference).norm()/reference.norm())
            info=dict(**stats,orthogonality_error=orth,before_objective=before,after_objective=after,
                      objective_gain=before-after,capture=capture,group_energy=energy,
                      objective_replay=replay_objective,executor_replay=replay,
                      price=price(updated,original))
            arms[name]=info;artifacts[label][name]=updated
            print(json.dumps(dict(label=label,arm=name,**{k:v for k,v in info.items() if k not in ('history','price')})),flush=True)
        gap=arms['original']['capture']-arms['graph']['capture']
        row=dict(label=label,arms=arms,matched_capture_gap=gap,
                 pred_a=check['held'] and all(v['converged'] and max(v['executor_replay'],v['objective_replay'])<=1e-9 for v in arms.values()),
                 pred_b=arms['graph']['objective_gain']>=1e-5,
                 pred_c=gap<=.001 and arms['graph']['price']['saved_float_fraction']>=.01)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='arms'}),flush=True)
    # Keep fitted graphs only; originals are reproducible from the same solver.
    torch.save({k:v['graph'] for k,v in artifacts.items()},root/'LL1_JOINT_CORE_SOLVE_V1_GRAPHS.pt')
    (root/'LL1_JOINT_CORE_SOLVE_V1_AUDIT.json').write_text(json.dumps(dict(control=check,rows=rows,
        scope='Strictly convex conditional core solves at fixed input spans and writers, compared to identically optimized original groups. No global graph/readers/writers convergence or behavioral circuit claim.'),indent=2)+'\n')


if __name__=='__main__':main()
