"""Paired structural effect geometry; opened outcomes, no operator selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
from mixed_state_projector_v1 import projector


def cosine(a,b):
    return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))


def vectors(report,q):
    foil=1 if report['foil']=='himself' else 2
    z=np.asarray(report['native_logits']);edited=np.asarray(report['arm_logits']['mixed'])
    native=q@(z[:,0]-z[:,foil])
    effect=q@((z[:,0]-z[:,foil])-(edited[:,0]-edited[:,foil]))
    return native,effect


def main():
    p=Path(__file__).resolve().parent
    paths=[p/'THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_RESULT.json',p/'THIRD_NOUN_WRITE_STRUCTURE_V1_RESULT.json']
    parent,current=[json.loads(path.read_text()) for path in paths]
    assert all(x['predictions']['pred_a_instrument'] for x in (parent,current))
    assert parent['corners']==current['corners']
    q=projector(current['corners'],2,4)
    reference={r['world_id']:vectors(r,q) for r in parent['reports']}
    reports=[]
    for r in current['reports']:
        wid=r['world_id'].split(':',1)[1];n0,e0=reference[wid];n,e=vectors(r,q)
        relative=float(np.linalg.norm(e)/np.linalg.norm(n));alignment=cosine(e,n)
        projection=float(e@n/(n@n))
        error=abs(projection-relative*alignment)
        assert error<1e-12 and abs(projection-r['metrics']['mixed']['live_natural_mixed_projection'])<1e-12
        reports.append({'world_id':r['world_id'],'layout':r['layout'],
            'effect_to_natural_norm_ratio':relative,'effect_natural_cosine':alignment,
            'projection':projection,'projection_identity_error':error,
            'native_original_cosine':cosine(n,n0),'effect_original_cosine':cosine(e,e0),
            'native_norm_relative_original':float(np.linalg.norm(n)/np.linalg.norm(n0)),
            'effect_norm_relative_original':float(np.linalg.norm(e)/np.linalg.norm(e0)),
            'failed_native_floor':r['native_mixed_rms']<.05,
            'failed_materiality':projection<.05,
            'failed_factor_spill':r['metrics']['mixed']['nonmixed_to_mixed_margin_ratio']>.25})
    result={'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reports':reports,'scope':'Paired output geometry after opening outcomes. No causal attribution to a producer or reader stage, no fitted correction, and no threshold rescue.'}
    with (p/'THIRD_NOUN_STRUCTURE_GEOMETRY_V1_RESULT.json').open('x') as handle:
        json.dump(result,handle,indent=2);handle.write('\n')
    for r in reports:
        print(r['world_id'],{k:round(r[k],4) for k in ('effect_to_natural_norm_ratio','effect_natural_cosine','native_original_cosine','effect_original_cosine')})


if __name__=='__main__':main()
