"""A exhaustive restricted optimum/function/node-intervention replay<=1e-10.
B planted shared graph uses fewer scalar products than3perquartic.
"""
import itertools,json
from pathlib import Path
import torch
from quartic_pair_dag_v1 import compile_pairs,execute,options
from sparse_quartic_core_v1 import indices


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(11520)
    out=Path(__file__).with_name('QUARTIC_PAIR_DAG_V1_CONTROL.json');assert not out.exists()
    terms=torch.tensor([[0,0,1,1],[0,0,1,2],[0,1,1,2],[0,1,2,2],[1,1,2,2]]).T
    allterms,allm=indices(3);m=torch.stack([allm[((allterms==t[:,None]).all(0)).nonzero().item()] for t in terms.T])
    dag=compile_pairs(terms)
    brute=min(len({p for c in cs for p in c}) for cs in itertools.product(*[options(t) for t in terms.T.tolist()]))
    assert dag['optimal'] and dag['pair_count']==brute and abs(dag['dual_bound']-brute)<1e-8
    reads=torch.randn(11,3);writer=torch.randn(7,5)
    direct=(reads[...,terms].prod(-2)*m)@writer.T;value=execute(dag,reads,m,writer)
    errors=[float((direct-value).norm()/direct.norm())]
    for node in range(dag['pair_count']):
        incident=(dag['children']==node).any(0).nonzero().flatten().tolist()
        a=execute(dag,reads,m,writer,zero_nodes=(node,));b=execute(dag,reads,m,writer,zero_edges=incident)
        errors.append(float((a-b).norm()/direct.norm()))
    nodes=(0,1);a=execute(dag,reads,m,writer,zero_nodes=(0,));b=execute(dag,reads,m,writer,zero_nodes=(1,));ab=execute(dag,reads,m,writer,zero_nodes=nodes)
    overlap=((dag['children']==0).any(0)&(dag['children']==1).any(0)).nonzero().flatten().tolist()
    onlyoverlap=value-execute(dag,reads,m,writer,zero_edges=overlap)
    errors.append(float((value-a-b+ab-onlyoverlap).norm()/direct.norm()))
    assert onlyoverlap.norm()>1e-6
    result=dict(pred_a=max(errors)<=1e-10,pred_b=dag['scalar_products']<dag['independent_scalar_products'],maximum_replay_error=max(errors),
        exhaustive_minimum_pairs=brute,dag={k:v.tolist() if torch.is_tensor(v) else v for k,v in dag.items()},
        scope='Exact minimum only among fixedmonomial 2+2 pair schedules, no distributive rewrites or semantic circuit proof.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a'] and result['pred_b']


if __name__=='__main__':main()
