"""CPU closure, saved predictor replay, and individual curvature omission tests."""
import json
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).parent;A=P.parent/'bilinear_quotient/circuits/followups'
def main():
    torch.set_num_threads(2);package=torch.load(A/'native_source_curvature_sum_v1.pt',map_location='cpu',weights_only=True)
    receipt=json.loads((A/'native_source_curvature_sum_v1_result.json').read_text());assert receipt['predictions']['pred_a_chain_closure']
    lookup={(r['panel'],r['role'],r['family'],r['arm']):r for r in receipt['records'] if r['mode']=='full'}
    names=list(package['cases'][0]['terms']);modes=['full']+['omit_'+n for n in names];scores={k:[] for k in modes};closures=[];replays=[];defects={n:[] for n in names}
    for case in package['cases']:
        terms={k:v.numpy() for k,v in case['terms'].items()};H=sum(terms.values());G=case['gradient'].numpy();reference=case['hessian_reference'].numpy();closures.append(float(np.max(abs(H-reference))))
        for n in names:defects[n].append(float(np.linalg.norm(terms[n])/max(np.linalg.norm(H),1e-30)))
        for mode in modes:
            modified=H if mode=='full' else H-terms[mode[5:]]
            for arm,av in case['amplitudes'].items():
                av=av.numpy();pred=-np.einsum('bop,bp->bo',G,av)-.5*np.einsum('bp,bopq,bq->bo',av,modified,av)
                for family in dict.fromkeys(case['families']):
                    ids=[i for i,f in enumerate(case['families']) if f==family];r=lookup[(case['panel'],case['role'],family,arm)];y=np.array(r['target']);den=max(np.linalg.norm(y[:,0]),1e-30)
                    if mode=='full':replays.append(float(np.max(abs(pred[ids]-r['prediction']))))
                    error=np.linalg.norm(pred[ids]-y,axis=0)/den;scores[mode].append(dict(panel=case['panel'],role=case['role'],family=family,arm=arm,number_error=float(error[0]),modal_error=error[1:].tolist()))
    assert max(closures)<1e-8 and max(replays)<1e-10
    summary={mode:dict(number_error=max(c['number_error'] for c in cs),modal_error=max(max(c['modal_error']) for c in cs),passes=all(c['number_error']<=.1 and max(c['modal_error'])<=.05 for c in cs)) for mode,cs in scores.items()}
    out=dict(cpu_hessian_closure_max_abs=max(closures),cpu_prediction_replay_max_abs=max(replays),summary=summary,maximum_coefficient_defect={k:max(v) for k,v in defects.items()},scores=scores,scope='Opened-data derivative-path necessity screen; leave-one-out successes do not imply joint omission success. Full native primal and first-order propagation retained.')
    (P/'SOURCE_CURVATURE_TERMS_CPU_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
