"""Fixed conventional response baselines on newly opened OOD receipts."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    r=json.loads((A/'source_ood_v1_result.json').read_text());lookup={(c['panel'],c['role'],c['family'],c['arm']):c for c in r['records']};scores=[];replay=[]
    for c in r['contexts']:
        G=np.array(c['gradient']);H=np.array(c['hessian']);w,V=np.linalg.eigh(H);order=np.argsort(-abs(w),axis=-1);w=np.take_along_axis(w,order,axis=-1);V=np.take_along_axis(V,order[...,None,:],axis=-1);rank2=np.einsum('boik,bok,bojk->boij',V[...,:2],w[...,:2],V[...,:2])
        rows=json.loads((P/f"SOURCE_OOD_V1_{c['panel'].upper()}_ROWS.json").read_text());rows=[x for x in rows if x['template']==c['template']]
        for arm,spec in r['plan']['amplitudes'].items():
            a=np.broadcast_to(spec,(len(G),5)) if spec is not None else np.array(c['candidate_amplitudes'])*(.5 if arm=='null_half' else 1.)
            for mode,T in [('full',H),('spectral2',rank2),('linear',np.zeros_like(H))]:
                pred=-np.einsum('bop,bp->bo',G,a)-.5*np.einsum('bp,bopq,bq->bo',a,T,a)
                for family in dict.fromkeys(x['family'] for x in rows):
                    ids=[i for i,x in enumerate(rows) if x['family']==family];row=lookup[(c['panel'],c['role'],family,arm)];y=np.array(row['target']);e=np.linalg.norm(pred[ids]-y,axis=0)/row['number_norm']
                    if mode=='full':replay.append(float(abs(pred[ids]-np.array(row['quadratic'])).max()))
                    scores.append(dict(mode=mode,panel=c['panel'],role=c['role'],family=family,arm=arm,number_error=float(e[0]),modal_error=float(max(e[1:]))))
    assert max(replay)<1e-12
    summary={mode:dict(number_error=max(c['number_error'] for c in scores if c['mode']==mode),modal_error=max(c['modal_error'] for c in scores if c['mode']==mode),passes=all(c['number_error']<=.1 and c['modal_error']<=.05 for c in scores if c['mode']==mode)) for mode in ['full','spectral2','linear']}
    out=dict(summary=summary,full_replay=max(replay),literal_context_values=dict(full=80,spectral2=68,linear=20),scope='Posthoc comparison of pre-existing fixed baseline rules on now-opened OOD outcomes; no new rank selection or finite-label fitting. Full native source/derivative generators retained.',scores=scores)
    (P/'SOURCE_OOD_BASELINES_V1_RESULT.json').write_text(json.dumps(out,separators=(',',':'))+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
