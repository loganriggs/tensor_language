"""Audit whether discarded initial-state response effects compose additively."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/subject_attention_freeze_v685_result.json'
OUT=Path(__file__).with_name('DISCARDED_RESPONSE_COMPOSITION_2026-09-20.json')

def main():
    r=json.loads(SOURCE.read_text())['initial_reports'];cells={};all_lost=[];all_cross=[]
    for name in r['sum']:
        def values(source):
            c=r[source][name]
            true=np.column_stack([c['target'],c['native_modal']])
            approx=np.column_stack([c['unprojected_prediction'],c['unprojected_modal']])
            return true,true-approx
        a,la=values('A');b,lb=values('B');s,ls=values('sum')
        cross=ls-la-lb
        all_lost.append(ls);all_cross.append(cross)
        cells[name]=dict(lost_sum_norm=np.linalg.norm(ls,axis=0).tolist(),
            additive_error_over_lost_sum=(np.linalg.norm(cross,axis=0)/np.maximum(np.linalg.norm(ls,axis=0),1e-20)).tolist(),
            additive_error_over_sum_target=(np.linalg.norm(cross,axis=0)/np.linalg.norm(s[:,0])).tolist(),
            additive_error_over_smaller_target=(np.linalg.norm(cross,axis=0)/min(np.linalg.norm(a[:,0]),np.linalg.norm(b[:,0]))).tolist(),
            lost_A=la.tolist(),lost_B=lb.tolist(),lost_sum=ls.tolist(),cross=cross.tolist())
    relative=np.linalg.norm(np.concatenate(all_cross),axis=0)/np.maximum(np.linalg.norm(np.concatenate(all_lost),axis=0),1e-20)
    result=dict(observables=['number','can/will','may/might','should/could'],
        prediction='lost(sum) versus lost(A)+lost(B) relativeL2<=.20 per observable pooled',
        pooled_additive_error_over_lost_sum=relative.tolist(),passes=bool(np.all(relative<=.20)),cells=cells,
        scope='Opened source interventions and native reconstruction control; output residual additivity is not feature identification or an executable correction.')
    OUT.write_text(json.dumps(result,indent=2)+'\n');print(relative, 'pass',result['passes'])
if __name__=='__main__':main()
