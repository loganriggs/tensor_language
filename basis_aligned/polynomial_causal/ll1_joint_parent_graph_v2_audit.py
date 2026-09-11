"""Score V2 against frozen V1 bars and diagnose joint parent span conflicts."""
import hashlib
import json
from pathlib import Path
import torch
import ll1_joint_parent_graph_v1 as v1
import ll1_joint_parent_graph_v2 as v2
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner
from chunked_bilinear_coefficient_v1 import dense


def additions(graph):
    d = graph['readers'].shape[1]
    count = len(graph['readers'])*(d-1)
    for group in graph['groups']:
        k, l, n = len(group['parent_ids']), len(group['lam']), len(group['shared_coeff'])
        count += l*(d-1)+max(l-1,0)+max(n-1,0)+k*max(l-1,0)+max(k-1,0)+2+len(group['writer'])
    return count


def mixed_control():
    # An exact mixed product: the private/private quadratic is identically zero.
    # A signed-square representation can make QR's null completion compete with
    # the required private reader in V1's degenerate zero spectrum.
    torch.manual_seed(2603)
    d=9; frame=torch.linalg.qr(torch.randn(d,d)).Q
    u,v,w=frame[:,:3].T
    a=torch.stack((torch.stack(((u+v)/2**.5,(u-v)/2**.5)),
                   torch.stack(((u+w)/2**.5,(u-w)/2**.5))))
    s=torch.tensor([[1.,-1.],[1.,-1.]])
    c=torch.eye(2); readers=u[None]; nodes=[dict(consumers=[0,1])]
    target=dense(*cp(a,s,c)); rows=[]
    for name, method in [('private_restriction',v1),('full_marginal',v2)]:
        graph=method.build(a,s,c,readers,nodes)
        actual=dense(*cp(*method.factors(graph)))
        err=float((actual-target).norm()/target.norm())
        x=torch.randn(13,d); reference=torch.einsum('oij,ni,nj->no',target,x,x)
        replay=float((method.execute(graph,x)-reference).norm()/reference.norm())
        rows.append(dict(method=name,coefficient_error=err,executor_error=replay))
    q=dense(*cp(a,s,c))[0]
    p=torch.eye(d)-u[:,None]*u[None]
    return dict(rows=rows,private_restriction_norm=float((p@q@p).norm()),
                private_marginal_norm=float((p@q@q@p).norm()),
                v2_held=max(rows[1]['coefficient_error'],rows[1]['executor_error'])<=1e-9,
                scope='Zero private/private energy is analytically degenerate. A numerical V1 success on this seed would not remove that ambiguity.')


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    control=mixed_control();print(json.dumps(dict(control=control)),flush=True)
    old=json.loads((root/'LL1_JOINT_PARENT_GRAPH_V1_AUDIT.json').read_text())
    proposals=torch.load(root/'LL1_SUBSPACE_PARENTS_V1_PROPOSALS.pt',weights_only=True)
    checkpoint=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin')
    sd=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    left,right,down=(sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down'))
    rows=[];graphs={};total=99245061353.47293
    for label in ('spectral','native'):
        source=Path(f'/dev/shm/bilin18_matched_shared_groups_v1_ll1_{label}.pt')
        saved=torch.load(source,weights_only=True,map_location='cpu')
        original=tuple(v.double() for v in saved['parts'])
        graph=v2.build(*original,proposals[label]['readers'],proposals[label]['nodes'])
        graph.update(output_whitener=saved['output_whitener'].double(),
                     bias=sd['transformer.h.17.mlp.Down_bias'].clone(),
                     source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        graphs[label]=graph
        candidate=cp(*v2.factors(graph));before=cp(*original)
        target=(left,right,saved['output_whitener'].double()@down)
        delta=(torch.cat((candidate[0],before[0])),torch.cat((candidate[1],before[1])),torch.cat((candidate[2],-before[2]),1))
        change=float(inner(delta,delta)/total)
        capture=float((2*inner(target,candidate)-inner(candidate,candidate))/total)
        prior=next(r for r in old['rows'] if r['label']==label)
        loss=prior['before_capture']-capture
        torch.manual_seed(2604);x=torch.randn(13,original[0].shape[-1])
        reference=((x@candidate[0].T)*(x@candidate[1].T))@candidate[2].T
        replay=float((v2.execute(graph,x)-reference).norm()/reference.norm())
        price=v2.price(graph,original);price['graph_scalar_additions']=additions(graph)
        a,s,c=original
        price['old_scalar_additions']=s.numel()*(a.shape[-1]-1)+len(s)*(s.shape[1]-1)+c.shape[1]*(len(c)-1)
        affected=[i for i,g in enumerate(graph['groups']) if len(g['parent_ids'])]
        membership=graph['minimum_joint_memberships']
        row=dict(label=label,price=price,executor_error=replay,squared_change_over_native=change,
                 graph_capture=capture,capture_loss=loss,v1_capture_loss=prior['capture_loss'],
                 loss_reduction_fraction=1-loss/prior['capture_loss'],
                 minimum_joint_membership=min(membership),
                 affected_groups=len(affected),groups_below_090_joint_membership=sum(membership[g]<.9 for g in affected),
                 group_memberships=[dict(group=g,parents=len(graph['groups'][g]['parent_ids']),minimum_squared_membership=membership[g]) for g in affected],
                 pred_a=replay<=1e-9 and control['v2_held'],
                 pred_b=loss<=.75*prior['capture_loss'],pred_c=loss<=.001)
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='group_memberships'}),flush=True)
    torch.save(graphs,root/'LL1_JOINT_PARENT_GRAPH_V2.pt')
    (root/'LL1_JOINT_PARENT_GRAPH_V2_AUDIT.json').write_text(json.dumps(dict(control=control,rows=rows,
        scope='Weights-only private-subspace proposal repair. Fixed parent incidence and parameter count; no continuous refitting, no behavioral identification, and V1 miss preserved.'),indent=2)+'\n')


if __name__=='__main__':main()
