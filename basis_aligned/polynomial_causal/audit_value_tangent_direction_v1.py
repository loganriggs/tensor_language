"""Best-scalar error floor from saved native tangent effect norms/projections."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'BILIN18_MLP4_VALUE_TANGENT_V1_RESULT.json'
OUT=ROOT/'VALUE_TANGENT_DIRECTION_V1_AUDIT.json'


if __name__=='__main__':
    r=json.loads(SOURCE.read_text());assert r['predictions']['pred_a_instrument'];cells={};closure=0.
    for panel,report in r['reports'].items():
        for side,c in report['cells'].items():
            fn=c['full_effect_norm'];pn=c['predicted_effect_norm'];projection=c['signed_effect_projection'];relative=c['relative_effect_error']
            ratio2=(pn/fn)**2;dot_scaled=projection
            closure=max(closure,abs(relative**2-(ratio2+1-2*dot_scaled)))
            cosine=dot_scaled/math.sqrt(ratio2)
            assert abs(cosine)<=1+1e-12
            floor=math.sqrt(max(0.,1-cosine*cosine))
            cells[panel+'__'+side]={'cosine':cosine,'best_possible_scalar_relative_error':floor,'observed_relative_error':relative,'oracle_scalar_for_bound_only':dot_scaled/ratio2}
    assert closure<1e-12
    result={'passed':True,'identity_max_error':closure,'cells':cells,'all_scalar_floors_exceed_registered_point_one':all(c['best_possible_scalar_relative_error']>.10 for c in cells.values()),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'model_forwards':0,'scope':'Exact finite-cohort best scalar bound for each aggregate centered-logit effect vector. Allows an oracle scalar perpanel/direction; no scalar is installed or promoted. Not a bound for a nonlinear/vector correction or future population.'}
    with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({'passed':True,'closure':closure,'floor_range':[min(c['best_possible_scalar_relative_error'] for c in cells.values()),max(c['best_possible_scalar_relative_error'] for c in cells.values())],'all_floors_exceed_point_one':result['all_scalar_floors_exceed_registered_point_one']}))
