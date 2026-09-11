"""Planted spectral/random initialization and reduced-objective red-team."""
import itertools
import json
import time
from pathlib import Path
import torch
from shared_reader_conditional_v1 import (
    partner_update, reader_update, group_cp, residual_cp, dense_cp,
)
from shared_reader_partner_rcg_v1 import optimize


def main():
    started=time.monotonic()
    torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
    torch.manual_seed(1401)
    dimension, outputs, rank=9,7,2
    rows=[]
    for count in [1,2]:
        planted=[]
        for _ in range(count):
            a=torch.randn(dimension);a/=a.norm()
            planted.append((a,torch.randn(outputs,rank),torch.randn(dimension,rank)))
        target=tuple(torch.cat([group_cp(*g)[k] for g in planted],dim=1 if k==2 else 0) for k in range(3))
        truth=dense_cp(*target); total=truth.square().sum()
        second=torch.einsum('oij,ojk->ik',truth,truth)
        spectral=torch.linalg.eigh(second).eigenvectors.flip(1)
        random=torch.randn(dimension,count);random/=random.norm(dim=0)
        for initialization, method in itertools.product(['spectral','random'],['alternating','reduced_rcg']):
            initial=spectral if initialization=='spectral' else random
            groups=[(initial[:,j],torch.zeros(outputs,rank),torch.zeros(dimension,rank)) for j in range(count)]
            previous=1.;increase=0.;evaluations=0;begin=time.monotonic()
            for sweep in range(200):
                for j in range(count):
                    residual=residual_cp(target,groups,j);a,_,_=groups[j]
                    if method=='reduced_rcg':
                        a,fit=optimize(a,*residual,torch.eye(outputs),total,rank,max_steps=30,tolerance=1e-9)
                        evaluations+=fit['evaluations']
                    u,v=partner_update(a,*residual,rank)
                    if method=='alternating':a,_=reader_update(*residual,u,v)
                    groups[j]=(a,u,v)
                    fitted=sum(dense_cp(*group_cp(*g)) for g in groups)
                    loss=float((truth-fitted).square().sum()/total)
                    increase=max(increase,loss-previous);previous=loss
                if loss<=1e-12:break
            actual=[dense_cp(*group_cp(*g)).flatten() for g in groups]
            native=[dense_cp(*group_cp(*g)).flatten() for g in planted]
            cos=torch.stack([torch.stack([x@y/(x.norm()*y.norm()) for y in native]) for x in actual])
            best=max(min(float(cos[j,p[j]]) for j in range(count)) for p in itertools.permutations(range(count)))
            rows.append(dict(groups=count,initialization=initialization,method=method,sweeps=sweep+1,
                             final_error=loss,maximum_increase=increase,matched_minimum_function_cosine=best,
                             evaluations=evaluations,seconds=time.monotonic()-begin))
            print(json.dumps(rows[-1]),flush=True)
    result=dict(predictions={
        'pred_a_monotone':all(r['maximum_increase']<=1e-9 for r in rows),
        'pred_b_spectral_reduced_recovery':all(r['final_error']<=1e-8 for r in rows if r['initialization']=='spectral' and r['method']=='reduced_rcg'),
        'pred_c_group_identification':all(r['matched_minimum_function_cosine']>=.99 for r in rows if r['final_error']<=1e-8)},
        rows=rows,seconds=time.monotonic()-started,
        scope='Independent planted problems; no retrospective repair to original random-start controls, no native fit/global guarantee.')
    Path(__file__).with_name('SHARED_READER_INITIALIZATION_V1_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['predictions']))


if __name__=='__main__':main()
