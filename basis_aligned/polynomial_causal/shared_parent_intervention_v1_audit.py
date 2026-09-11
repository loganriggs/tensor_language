"""Weight-only removal/composition accounting on frozen pre-fit native graphs."""
import json
from pathlib import Path
import torch
from shared_parent_intervention_v1 import disable,concatenate,subset,value,banks,group_parent
from ll1_joint_parent_graph_v2 import execute,factors
from symmetric_ll1_objective_v1 import cp
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def audit(graph,label):
    mixed,pair,nodes=banks(graph);pairs=graph['pairs'];rows=[];checks=[]
    torch.manual_seed(3301);x=torch.randn(13,graph['readers'].shape[1]);base=execute(graph,x)
    aa,ss,cc=factors(graph)
    bank=cp(aa,ss,cc);checks.append(float((base-value(bank,x)).norm()/base.norm()))
    group_energy=cc.square().sum(1)*((aa@aa.transpose(1,2)).square()*ss[:,:,None]*ss[:,None]).sum((1,2))
    for i,node in enumerate(nodes):
        effect=base-disable(graph,x,[i]);replay=float((effect-value(node,x)).norm()/effect.norm())
        checks.append(replay);consumers=[];sum_energy=0.
        for j,g in enumerate(graph['groups']):
            if i not in g['parent_ids'].tolist():continue
            branch=group_parent(graph,j,i);energy=float(inner(branch,branch));sum_energy+=energy
            consumers.append(dict(group=j,removal_energy_over_group=energy/float(group_energy[j])))
        energy=float(inner(node,node))
        rows.append(dict(parent=i,consumer_count=len(consumers),
                         consumers_above_one_percent=sum(c['removal_energy_over_group']>=.01 for c in consumers),
                         removal_energy=energy,sum_individual_consumer_energy=sum_energy,
                         combined_over_sum_consumer_energy=energy/sum_energy if sum_energy else None,
                         consumers=consumers,executor_replay=replay))
    pair_rows=[]
    for k,(i,j) in enumerate(pairs.tolist()):
        if i==j:continue
        term=subset(pair,torch.tensor([k]));negative=(term[0],term[1],-term[2])
        joint=concatenate(nodes[i],nodes[j],negative)
        energy=float(inner(joint,joint));correction=float(inner(term,term))
        pair_rows.append(dict(parents=[i,j],pair_index=k,pair_correction_energy=correction,
                              joint_removal_energy=energy,correction_over_joint=correction/energy if energy else None))
    if pair_rows:
        chosen=max(pair_rows,key=lambda r:r['pair_correction_energy'])
        i,j=chosen['parents'];term=subset(pair,torch.tensor([chosen['pair_index']]))
        predicted=value(nodes[i],x)+value(nodes[j],x)-value(term,x)
        actual=base-disable(graph,x,[i,j]);checks.append(float((actual-predicted).norm()/actual.norm()))
    all_bank=concatenate(mixed,pair);actual=base-disable(graph,x,range(len(nodes)))
    checks.append(float((actual-value(all_bank,x)).norm()/actual.norm()))
    naive=sum((value(node,x) for node in nodes),torch.zeros_like(actual))
    offdiag=subset(pair,pairs[:,0]!=pairs[:,1])
    corrected=naive-value(offdiag,x)
    checks.append(float((actual-corrected).norm()/actual.norm()))
    return dict(label=label,parents=rows,pairs=pair_rows,
                maximum_executor_replay=max(checks),
                naive_all_parent_sum_relative_output_error=float((actual-naive).norm()/actual.norm()),
                corrected_all_parent_sum_relative_output_error=float((actual-corrected).norm()/actual.norm()),
                pred_a=max(checks)<=1e-9,
                pred_b=any(r['correction_over_joint'] is not None and r['correction_over_joint']>=.01 for r in pair_rows),
                pred_c=all(r['consumers_above_one_percent']>=2 for r in rows),
                parents_with_two_effective_consumers=sum(r['consumers_above_one_percent']>=2 for r in rows))


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    root=Path(__file__).parent
    graphs=torch.load(root/'LL1_JOINT_CORE_SOLVE_V1_GRAPHS.pt',weights_only=True)
    rows=[]
    for label,graph in graphs.items():
        row=audit(graph,label);rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ('parents','pairs')}),flush=True)
    result=dict(rows=rows,scope='Exact intervention identities of the frozen approximate DAG, in full-U isometric coefficient coordinates. No native full-model or behavioral effect claim; graph-node deletion differs from deleting an input direction everywhere.')
    (root/'SHARED_PARENT_INTERVENTION_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
