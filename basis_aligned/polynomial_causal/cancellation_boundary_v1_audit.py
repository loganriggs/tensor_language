"""Graph closure versus matched-size expansions on two frozen fit snapshots.

A Gram/CP<=1e-8 and bounds; B graph128 cosine>=.99 and half base drift;
C graph128 drift<=half energy-control drift. No independent-start stability.
"""
import json,time
import torch
from cancellation_group_v1_audit import load
from native_support_exchange_v1_audit import P
from joint_quadratic_fit_v1 import product_cross
from structured_branch_amplitudes_v1 import inner


def closure(base,edges):
    selected=set(base)
    while True:
        expanded=selected|{v for e in edges if e['i'] in selected or e['j'] in selected for v in (e['i'],e['j'])}
        if expanded==selected:return sorted(selected)
        selected=expanded


def main():
    torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);torch.set_num_threads(2)
    out=P/'CANCELLATION_BOUNDARY_V1_AUDIT.json';assert not out.exists();start=time.perf_counter()
    audit=json.loads((P/'PROJECTED_PRODUCT_CANCELLATION_V1_AUDIT.json').read_text())
    older=json.loads((P/'NATIVE_SUPPORT_EXCHANGE_V1_AUDIT.json').read_text())['source']
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T;current=load(audit['source'],wh);previous=load(older,wh)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    a,b,w=current
    energy=w.square().sum(0)*.5*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())
    order=energy.argsort(descending=True).tolist();base=audit['top16_products'];memo={}
    def measure(ids):
        key=tuple(sorted(ids))
        if key in memo:return memo[key]
        group=(a[ids],b[ids],w[:,ids]);old=(previous[0][ids],previous[1][ids],previous[2][:,ids])
        new_energy=inner(group,group);old_energy=inner(old,old);cross=inner(group,old)
        gram=product_cross(group[0],group[1],group[0],group[1])*(group[2].T@group[2])
        replay=abs(float(gram.sum()-new_energy))/total
        row=dict(products=list(ids),count=len(ids),energy=float(new_energy/total),old_energy=float(old_energy/total),
            cosine=float(cross/(new_energy*old_energy).sqrt()),
            squared_function_change=float((new_energy+old_energy-2*cross)/total),gram_replay=replay)
        memo[key]=row;return row
    base_result=measure(base);rows=[]
    remaining=torch.tensor([j for j in range(4608) if j not in base])
    random_order=remaining[torch.randperm(len(remaining),generator=torch.Generator().manual_seed(772))].tolist()
    for edgecount in (32,64,128):
        ids=closure(base,audit['largest_negative_pairs'][:edgecount]);n=len(ids)
        for label,selected in [('graph',ids),('energy',order[:n]),('random',base+random_order[:n-len(base)])]:
            rows.append(dict(edge_count=edgecount,method=label,**measure(selected)))
    graph=next(r for r in rows if r['edge_count']==128 and r['method']=='graph')
    control=next(r for r in rows if r['edge_count']==128 and r['method']=='energy')
    numerical=all(r['gram_replay']<=1e-8 and abs(r['cosine'])<=1+1e-8 and r['squared_function_change']>=-1e-10
                  and r['count']<=256 for r in rows)
    result=dict(predictions=dict(pred_a_instrument=numerical,
        pred_b_stable_boundary=graph['cosine']>=.99 and graph['squared_function_change']<=.5*base_result['squared_function_change'],
        pred_c_graph_advantage=graph['squared_function_change']<=.5*control['squared_function_change']),
        base=base_result,rows=rows,source=audit['source'],previous_source=older,seconds=time.perf_counter()-start,
        scope='Fixed snapshot graph, within-run function drift only. Not fresh-start, behavioral or semantic circuit stability.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(predictions=result['predictions'],base={k:v for k,v in base_result.items() if k!='products'},
                         rows=[{k:v for k,v in r.items() if k!='products'} for r in rows],seconds=result['seconds']),indent=2))


if __name__=='__main__':main()
