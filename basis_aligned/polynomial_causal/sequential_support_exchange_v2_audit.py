"""Sequential native support exchanges; original independent gains are not summed as evidence.

pred_a global CP gain replay<=1e-9, finite monotone conditional gains;
pred_b achieved gain>=.8 times earlier independent sum; pred_c CPU compute<=60s.
"""
import hashlib
import json
import time
from pathlib import Path
import torch
from native_support_exchange_v1_audit import P, CK
from folded_sparse_dictionary_v1 import decode
from folded_support_exchange_v1 import quadratic, best_exchange
from structured_branch_amplitudes_v1 import inner


def main():
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    out=P/'SEQUENTIAL_SUPPORT_EXCHANGE_V2_AUDIT.json'; assert not out.exists()
    previous=json.loads((P/'NATIVE_SUPPORT_EXCHANGE_V1_AUDIT.json').read_text())
    source=Path(previous['source']['path'])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==previous['source']['sha256']
    saved=torch.load(source,weights_only=True,map_location='cpu')
    sd=torch.load(CK,weights_only=True,map_location='cpu',mmap=True)
    l,r,down=[sd[f'transformer.h.17.mlp.{k}.weight'].double() for k in ('Left','Right','Down')]
    metric=torch.load('/dev/shm/bilin18_native_product_energy_v1.pt',weights_only=True,map_location='cpu')['unembedding_gram'].double()
    wh=torch.linalg.cholesky(metric).T; native=(l,r,wh@down)
    ids=saved['code_indices'].long().clone(); values=saved['code_values'].double().clone()
    readers,basis,_,_=decode(saved['analysis_basis'].double(),ids,values,torch.ones(len(ids)))
    a,b=readers.chunk(2); w=wh@saved['down'].double(); original=(a.clone(),b.clone(),w)
    total=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())['native_total']
    started=time.perf_counter(); bg=basis@basis.T; rows=[]; deltas=[]
    for prior in previous['rows']:
        j=prior['product']; side=prior['side']
        aa,bb=(a,b) if side=='Left' else (b,a)
        nn=native if side=='Left' else (r,l,native[2]); index=j if side=='Left' else j+4608
        support=ids[index].tolist(); gram,rhs=quadratic(nn,aa,bb,w,j,basis,bg)
        old=values[index]; old_energy=old@gram[support][:,support]@old-2*old@rhs[support]
        new_ids,new_values,report=best_exchange(gram,rhs,support)
        new_energy=new_values@gram[new_ids][:,new_ids]@new_values-2*new_values@rhs[new_ids]
        gain=float((old_energy-new_energy)/total)
        new_reader=new_values@basis[new_ids]
        deltas.append(((new_reader-aa[j]).clone(),bb[j].clone(),w[:,j].clone()))
        aa[j]=new_reader; ids[index]=torch.tensor(new_ids); values[index]=new_values
        rows.append(dict(side=side,product=j,total_gain=gain,cardinality=len(set(new_ids)),**report))
    delta=(torch.stack([d[0] for d in deltas]),torch.stack([d[1] for d in deltas]),
           torch.stack([d[2] for d in deltas],dim=1))
    measured=-float((inner(delta,delta)+2*inner(original,delta)-2*inner(native,delta))/total)
    predicted=sum(row['total_gain'] for row in rows)
    independent=sum(row['independent_cp_total_gain'] for row in previous['rows'])
    seconds=time.perf_counter()-started
    cache=Path('/dev/shm/bilin18_sequential_support_exchange_v2.pt'); assert not cache.exists()
    torch.save(dict(analysis_basis=basis,code_indices=ids.to(torch.int16),code_values=values,
                    down=saved['down'],retained_bias_key=saved['retained_bias_key'],
                    source=previous['source'],loss=saved['loss']-measured),cache)
    result=dict(predictions=dict(pred_a_instrument=abs(measured-predicted)<=1e-9
                    and all(row['total_gain']>=-1e-12 and row['cardinality']==128 for row in rows)
                    and bool(torch.isfinite(torch.tensor(measured))),
                pred_b_retained=measured>=.8*independent,pred_c_cost=seconds<=60),
                actual_gain=measured,conditional_sum=predicted,independent_sum=independent,
                gain_ratio=measured/independent,replay_error=abs(measured-predicted),
                initial_capture=1-saved['loss'],final_capture=1-saved['loss']+measured,
                compute_seconds=seconds,rows=rows,source=previous['source'],
                cache=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest()),
                scope='32 sequential weight-space replacements, fixed features/output, no behavior or global support optimum.')
    with out.open('x') as f: json.dump(result,f,indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))


if __name__=='__main__': main()
