"""Frozen native comparison for compatible shared-parent incidence selection."""
import hashlib
import json
from pathlib import Path
import torch
from ll1_compatible_parents_v1 import select
from ll1_joint_parent_graph_v2 import build,execute,factors,price
from ll1_joint_parent_graph_v2_audit import additions
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    proposals=torch.load(root/'LL1_SUBSPACE_PARENTS_V1_PROPOSALS.pt',weights_only=True)
    old=json.loads((root/'LL1_JOINT_PARENT_GRAPH_V2_AUDIT.json').read_text())
    checkpoint=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
    sd=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    left,right,down=(sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down'))
    rows=[];graphs={};total=99245061353.47293
    for label in ('spectral','native'):
        source=Path(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt')
        saved=torch.load(source,weights_only=True,map_location='cpu')
        original=tuple(v.double() for v in saved['parts']);proposal=proposals[label]
        readers,nodes,selection=select(original[0],proposal['readers'],proposal['nodes'])
        graph=build(*original,readers,nodes)
        graph.update(output_whitener=saved['output_whitener'].double(),
                     bias=sd['transformer.h.17.mlp.Down_bias'].clone(),
                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),nodes=nodes)
        graphs[label]=graph
        candidate=cp(*factors(graph));before=cp(*original)
        target=(left,right,saved['output_whitener'].double()@down)
        delta=(torch.cat((candidate[0],before[0])),torch.cat((candidate[1],before[1])),torch.cat((candidate[2],-before[2]),1))
        change=float(inner(delta,delta)/total)
        loss=float((inner(delta,delta)+2*inner(before,delta)-2*inner(target,delta))/total)
        prior=next(r for r in old['rows'] if r['label']==label)
        torch.manual_seed(2701);x=torch.randn(13,original[0].shape[-1])
        reference=((x@candidate[0].T)*(x@candidate[1].T))@candidate[2].T
        replay=float((execute(graph,x)-reference).norm()/reference.norm())
        costs=price(graph,original);costs['graph_scalar_additions']=additions(graph)
        multi=sum(len(g['parent_ids'])>=2 for g in graph['groups'])
        row=dict(label=label,price=costs,retained_parents=len(readers),groups_with_multiple_parents=multi,
                 executor_error=replay,squared_change_over_native=change,capture_loss=loss,
                 v2_capture_loss=prior['capture_loss'],loss_reduction_fraction=1-loss/prior['capture_loss'],
                 retained_savings_fraction=costs['saved_float_fraction']/prior['price']['saved_float_fraction'],
                 minimum_joint_membership=min(graph['minimum_joint_memberships']),selection=selection,
                 pred_a=replay<=1e-9 and min(graph['minimum_joint_memberships'])>=.90-1e-12,
                 pred_b=loss<=.5*prior['capture_loss'] and costs['saved_float_fraction']>=.5*prior['price']['saved_float_fraction'],
                 pred_c=loss<=.001 and multi>=1)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='selection'}),flush=True)
    torch.save(graphs,root/'LL1_COMPATIBLE_PARENTS_V1.pt')
    (root/'LL1_COMPATIBLE_PARENTS_V1_AUDIT.json').write_text(json.dumps(dict(rows=rows,
        scope='Joint-compatible incidence selection on frozen unconverged pilots. Exact executor, approximate original function; no data fitting or circuit identification.'),indent=2)+'\n')


if __name__=='__main__':main()
