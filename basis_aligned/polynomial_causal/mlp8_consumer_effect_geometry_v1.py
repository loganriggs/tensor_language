"""Compare opened path/total effects; verify a source-by-mediator control.

A numerical difference is not an isolated alternate route without a joint intervention.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from mixed_state_projector_v1 import projector


def factorial_effects(outputs):
    # source state 0=native,1=edited; attention response 0=native,1=edited
    z=np.asarray(outputs);assert z.shape[:2]==(2,2)
    total=z[0,0]-z[1,1]
    attention=z[0,0]-z[0,1]
    bypass=z[0,0]-z[1,0]
    interaction=total-attention-bypass
    assert np.allclose(total,attention+bypass+interaction,rtol=1e-12,atol=1e-12)
    return total,attention,bypass,interaction


def controls():
    # At the joint source/attention interface, source changes x and attention changes a.
    def cube(alpha):return np.array([[x+a+alpha*x*a for a in (1.,0.)] for x in (1.,0.)])
    additive=factorial_effects(cube(0));coupled=factorial_effects(cube(2))
    assert additive==(2.,1.,1.,0.) and coupled==(4.,3.,3.,-2.)
    assert coupled[0]-coupled[1]!=coupled[2]
    return {'passed':True,'additive':list(additive),'coupled':list(coupled),
            'effect_order':['total','attention_at_native_source','bypass_at_native_attention','interaction'],
            'subtraction_is_not_native_background_bypass':True}


def main():
    p=Path(__file__).resolve().parent;paths=[p/'MATURE_VALUE_MLP8_CONSUMERS_V1_RESULT.json',p/'MATURE_VALUE_MLP8_ORIGIN_V1_RESULT.json']
    all_data,path_data=[json.loads(f.read_text()) for f in paths]
    assert all_data['predictions']['pred_a_instrument'] and path_data['predictions']['pred_a_instrument']
    lookup={r['world_id']:r for r in path_data['reports']};q=projector(all_data['corners'],2,4);reports=[]
    for row in all_data['reports']:
        old=lookup[row['world_id']];z0=np.asarray(row['native_logits']);foil=1 if row['foil']=='himself' else 2
        assert np.array_equal(z0,np.asarray(old['native_logits']))
        record={'world_id':row['world_id'],'layout':row['layout'],'arms':{}}
        for arm in ('full','new','inherited'):
            total=q@(z0-np.asarray(row['arm_logits'][arm]));path=q@(z0-np.asarray(old['arm_logits'][arm]));metrics={}
            for label,t,v in [('margin',total[:,0]-total[:,foil],path[:,0]-path[:,foil]),
                              ('readers',total-total.mean(-1,keepdims=True),path-path.mean(-1,keepdims=True))]:
                den=np.linalg.norm(t);vn=np.linalg.norm(v);assert den>0 and vn>0
                metrics[label]={'signed_projection':float(np.sum(t*v)/den**2),
                    'relative_norm':float(vn/den),'cosine':float(np.sum(t*v)/(den*vn)),
                    'relative_difference':float(np.linalg.norm(t-v)/den)}
            record['arms'][arm]=metrics
        reports.append(record)
    result={'controls':controls(),'reports':reports,'sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        'scope':'Opened path-versus-total geometry and synthetic source/attention factorial. Not native mediator identification.'}
    with (p/'MLP8_CONSUMER_EFFECT_GEOMETRY_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for layout in ('original','fronted_pp'):
        rows=[r for r in reports if r['layout']==layout]
        print(layout,{k:[min(r['arms']['full']['margin'][k] for r in rows),max(r['arms']['full']['margin'][k] for r in rows)] for k in ('signed_projection','relative_norm','cosine')})
    print(result['controls'])

if __name__=='__main__':main()
