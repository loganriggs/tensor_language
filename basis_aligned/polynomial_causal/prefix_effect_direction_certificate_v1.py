"""One-effect-axis obstruction from measured residual and effect norms."""
import hashlib,json
from pathlib import Path
import numpy as np


def certificate(effect_norm,remaining_norm):
    m,r=effect_norm,remaining_norm
    if m is None or r is None or m==0:return None
    projection=(1+m*m-r*r)/2
    cosine=projection/m
    assert abs(cosine)<=1+1e-10
    return {'signed_projection_on_native':projection,'cosine_with_native':cosine,
            'minimum_error_on_entire_signed_effect_axis':float(np.sqrt(max(0.,1-cosine*cosine)))}


def main(source,target):
    worst=0.;rng=np.random.default_rng(9100948)
    for _ in range(32):
        native=rng.normal(size=17);effect=rng.normal(size=17);scale=np.linalg.norm(native)
        c=certificate(np.linalg.norm(effect)/scale,np.linalg.norm(native-effect)/scale)
        coefficient=np.vdot(native,effect)/np.vdot(effect,effect)
        exact=np.linalg.norm(native-coefficient*effect)/scale
        worst=max(worst,abs(exact-c['minimum_error_on_entire_signed_effect_axis']))
    assert worst<1e-12
    parent=json.loads(source.read_text());assert parent['predictions']['pred_a_instrument']
    reports=[]
    for row in parent['reports']:
        reports.append({'world_id':row['world_id'],'layout':row['layout'],
                        'measurements':{metric:certificate(row['materiality'][metric],row['necessity_errors'][metric])
                                        for metric in ('margin','readers','vocabulary')}})
    names=['signed_projection_on_native','cosine_with_native','minimum_error_on_entire_signed_effect_axis']
    ranges={metric:{name:[min(r['measurements'][metric][name] for r in reports),max(r['measurements'][metric][name] for r in reports)]
                    for name in names} for metric in ('margin','readers','vocabulary')}
    result={'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reports':reports,'ranges':ranges,
            'controls':{'passed':True,'random_fixtures':32,'projection_identity_max_error':worst},
            'scope':'Geometric certificate for the single measured complete-prefix-removal effect axis, relative to native Q_oh output signal. No native gain intervention or fit is executed. Does not rule out another nonlinear program, a different effect direction, or all uses of prefix memory.'}
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]),Path(sys.argv[2]))
