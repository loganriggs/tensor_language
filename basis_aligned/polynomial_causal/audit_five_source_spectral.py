"""Same-interface per-context signed spectral baseline; CPU, opened data only."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    torch.set_num_threads(2)
    package=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)
    native=json.loads((A/'five_source_full_span_v1_result.json').read_text())
    lookup={(r['panel'],r['role'],r['family'],tuple(r['source_set'])):r for r in native['records'] if r['mode']=='full'}
    scores={str(k):[] for k in range(1,6)};closure=[];replays=[]
    other=json.loads((A/'native_source_curvature_sum_v1_result.json').read_text())
    other_lookup={(r['panel'],r['role'],r['family'],r['arm']):r for r in other['records'] if r['mode']=='full'}
    other_scores={str(k):[] for k in range(1,6)}
    for case in package['cases']:
        H=case['hessian_reference'].numpy();G=case['gradient'].numpy()
        lam,V=np.linalg.eigh(H);order=np.argsort(-abs(lam),axis=-1)
        lam=np.take_along_axis(lam,order,axis=-1);V=np.take_along_axis(V,order[...,None,:],axis=-1)
        for rank in range(1,6):
            R=np.einsum('boik,bok,bojk->boij',V[...,:rank],lam[...,:rank],V[...,:rank])
            if rank==5:closure.append(float(abs(R-H).max()))
            for arm,av in case['amplitudes'].items():
                av=av.numpy();pred=-np.einsum('bop,bp->bo',G,av)-.5*np.einsum('bp,bopq,bq->bo',av,R,av)
                for family in dict.fromkeys(case['families']):
                    ids=[i for i,f in enumerate(case['families']) if f==family]
                    r=other_lookup[(case['panel'],case['role'],family,arm)];y=np.array(r['target'])
                    errors=np.linalg.norm(pred[ids]-y,axis=0)/max(np.linalg.norm(y[:,0]),1e-30)
                    other_scores[str(rank)].append(dict(panel=case['panel'],role=case['role'],family=family,arm=arm,number_error=float(errors[0]),modal_error=float(max(errors[1:]))))
            for selected in native['plan']['source_sets']:
                a=np.zeros(5);a[selected]=1
                pred=-np.einsum('bop,p->bo',G,a)-.5*np.einsum('p,bopq,q->bo',a,R,a)
                for family in dict.fromkeys(case['families']):
                    ids=[i for i,f in enumerate(case['families']) if f==family]
                    r=lookup[(case['panel'],case['role'],family,tuple(selected))]
                    errors=np.linalg.norm(pred[ids]-np.array(r['target']),axis=0)/r['number_norm']
                    if rank==5:replays.append(float(abs(pred[ids]-np.array(r['prediction'])).max()))
                    scores[str(rank)].append(dict(panel=case['panel'],role=case['role'],family=family,source_set=selected,number_error=float(errors[0]),modal_error=float(max(errors[1:]))))
    assert max(closure)<1e-12 and max(replays)<1e-12
    summary={k:dict(max_number_error=max(r['number_error'] for r in rows),max_modal_error=max(r['modal_error'] for r in rows),passes=all(r['number_error']<=.1 and r['modal_error']<=.05 for r in rows),literal_values_per_context=20+4*6*int(k)) for k,rows in scores.items()}
    other_summary={k:dict(max_number_error=max(r['number_error'] for r in rows),max_modal_error=max(r['modal_error'] for r in rows),passes=all(r['number_error']<=.1 and r['modal_error']<=.05 for r in rows)) for k,rows in other_scores.items()}
    out=dict(summary=summary,unit_and_modal_null_summary=other_summary,full_rank_coefficient_replay=max(closure),full_rank_prediction_replay=max(replays),dense_symmetric_values_per_context=80,scores=scores,scope='Per-context/output signed spectral truncation of analytic Hessians. Same full native contexts and exact gradients retained. Opened finite single/pair directions; not OOD, shared features, HT or a causal intervention. Literal factor storage includes five-vector plus eigenvalue per rank/output, without geometric parameter discounts.')
    (P/'FIVE_SOURCE_SPECTRAL_BASELINE_CPU_RESULT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='scores'},indent=2))
if __name__=='__main__':main()
