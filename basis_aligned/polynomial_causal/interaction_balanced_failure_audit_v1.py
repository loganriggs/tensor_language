"""Descriptive paired failure audit; registered native bars stay unchanged."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def main():
    rows=[]
    baseline=None
    for name,prefix in [('ordinary','INTERACTION_SHARED_WRITE_REGIONAL_V1'),('balanced','INTERACTION_BALANCED_REGIONAL_V1')]:
        r=json.loads((P/(prefix+'_RESULT.json')).read_text())
        ref=np.array(r['reference_own_effects']);pred=np.array(r['predicted_own_effects'])
        if baseline is None:baseline=ref
        assert np.max(np.abs(ref-baseline))<1e-10
        for group in range(5):
            rr=ref[group*24:(group+1)*24];ee=pred[group*24:(group+1)*24]-rr
            material=np.abs(rr)>=1e-5
            loo=[]
            for pair in range(12):
                keep=np.ones(24,dtype=bool);keep[2*pair:2*pair+2]=False
                loo.append(float(np.linalg.norm(ee[keep])/np.linalg.norm(rr[keep])))
            rows.append(dict(method=name,group=group,own_error=float(np.linalg.norm(ee)/np.linalg.norm(rr)),
                material_rows=int(material.sum()),material_error=float(np.linalg.norm(ee[material])/np.linalg.norm(rr[material])),
                leave_one_pair_out_min=min(loo),leave_one_pair_out_max=max(loo),
                largest_pair_error_energy_share=float((ee.reshape(12,2)**2).sum(1).max()/(ee**2).sum())))
    out=dict(rows=rows,scope='Descriptive fixed material threshold1e-5 and leave-one-contiguous-cue-pair-out. Original all-row 10% bars unchanged; no statistical confidence, refitting or OOD evidence.')
    (P/'INTERACTION_BALANCED_FAILURE_V1_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
