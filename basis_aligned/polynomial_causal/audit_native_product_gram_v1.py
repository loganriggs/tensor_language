"""Exact CPU Gram of native folded product functions; not a reuse lower bound."""
import json
from pathlib import Path
import time
import torch
from audit_native_reader_metric_v1 import P,CK,digest


@torch.no_grad()
def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(2)
    started=time.perf_counter()
    source=json.loads((P/'NATIVE_READER_METRIC_V1_AUDIT.json').read_text())
    assert digest(source['cache']['path'])==source['cache']['sha256']
    cached=torch.load(source['cache']['path'],map_location='cpu',weights_only=True)
    sd=torch.load(CK,map_location='cpu',mmap=True,weights_only=True)
    l,r,d=[sd[f'transformer.h.17.mlp.{key}.weight'].double() for key in ('Left','Right','Down')]
    e=cached['product_energy'];gd=cached['unembedding_gram']@d;n=len(e)
    squares=0.;normalized_squares=0.;total=0.;diag_error=0.;largest=[]
    counts={'.1':0,'.5':0}
    for start in range(0,n,256):
        stop=min(start+256,n);sl=slice(start,stop)
        forms=((l[sl]@l.T)*(r[sl]@r.T)+(l[sl]@r.T)*(r[sl]@l.T))/2
        h=(d[:,sl].T@gd)*forms
        rows=torch.arange(stop-start);cols=torch.arange(start,stop)
        diag_error=max(diag_error,float(((h[rows,cols]-e[sl])/e[sl]).abs().max()))
        c=h/(e[sl,None]*e[None,:]).sqrt()
        squares+=float(h.square().sum());normalized_squares+=float(c.square().sum());total+=float(h.sum())
        c[rows,cols]=0.
        for key in counts:counts[key]+=int((c.abs()>float(key)).sum())
        values,indices=c.abs().reshape(-1).topk(10)
        for value,index in zip(values.tolist(),indices.tolist()):
            i,j=start+index//n,index%n
            largest.append(dict(i=min(i,j),j=max(i,j),absolute_correlation=value))
    unique={}
    for row in largest:unique[(row['i'],row['j'])]=row
    pairs=sorted(unique.values(),key=lambda r:r['absolute_correlation'],reverse=True)[:20]
    effective=n*n/normalized_squares
    replay=abs(total-source['native_total'])/source['native_total']
    result=dict(predictions={
        'pred_a_exact_gram':diag_error<=1e-10 and replay<=1e-8,
        'pred_b_nearly_full_effective_rank':effective>=.9*n,
        'pred_c_no_highly_correlated_pair':pairs[0]['absolute_correlation']<=.5},
        product_count=n,normalized_gram_effective_rank=effective,
        energy_weighted_gram_effective_rank=float(e.sum().square())/squares,
        diagonal_relative_error=diag_error,total_relative_error=replay,
        sum_gram=total,offdiagonal_pair_fractions={k:v/(n*(n-1)) for k,v in counts.items()},
        largest_pairs=pairs,seconds=time.perf_counter()-started,
        source_sha256=digest(__file__),metric_cache=source['cache'],
        gpu_access=False,corpus_access=False,
        scope='Correlation of existing complete product functions only. Orthogonal components may share input features; no bound on general tensor refactorization or arithmetic reuse.')
    with (P/'NATIVE_PRODUCT_GRAM_V1_AUDIT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('largest_pairs','metric_cache')},indent=2))
    print(json.dumps(pairs[:5]))


if __name__=='__main__':main()
