"""Same source displacement, two consumers: context-free effect obstruction."""
import hashlib,json
from pathlib import Path
import numpy as np
import mixed_state_projector_v1 as Q


def differences(e):
    return e[1,0]-e[0,0],e[1,1]-e[0,1]


def controls():
    rng=np.random.default_rng(9100910);error=0.
    for _ in range(32):
        e=rng.normal(size=(2,2,9));offset=rng.normal(size=(2,9))
        a,b=differences(e);aa,bb=differences(e-offset[None])
        midpoint=(a+b)/2;bound=np.linalg.norm(b-a)/2
        error=max(error,float(np.max(np.abs(a-aa))),float(np.max(np.abs(b-bb))),
                  abs(np.linalg.norm(a-midpoint)-bound),abs(np.linalg.norm(b-midpoint)-bound))
    # One shared nonlinear operation with explicit context, no different algorithms.
    sources=np.array([1.,2.]);backgrounds=np.array([0.,3.])
    e=(sources[:,None]+backgrounds[None,:])**2-backgrounds[None,:]**2
    a,b=differences(e)
    assert (a,b)==(3.,9.) and b-a==2*(backgrounds[1]-backgrounds[0])*(sources[1]-sources[0])
    assert error<1e-12
    return {'passed':True,'random_fixtures':32,'offset_and_midpoint_max_error':error,
            'shared_square':{'effect_table':e.tolist(),'same_source_edit_effects':[float(a),float(b)],
                             'interaction':float(b-a),'common_effect_minimax_error':float(abs(b-a)/2)}}


def main(source,target):
    parent=json.loads(source.read_text());assert parent['predictions']['pred_a_instrument']
    q=Q.projector(parent['corners'],2,4);reports=[]
    for r in parent['reports']:
        e=np.einsum('ij,abjv->abiv',q,np.asarray(r['effect_logits']))
        a,b=differences(e);foil=1 if r['foil']=='himself' else 2
        def view(x,kind):
            return x[:,0]-x[:,foil] if kind=='margin' else x-x.mean(-1,keepdims=True)
        metrics={}
        for kind in ('margin','readers'):
            av,bv=view(a,kind),view(b,kind);den=max(np.linalg.norm(av),np.linalg.norm(bv))
            mismatch=np.linalg.norm(bv-av)
            metrics[kind]={'edit_norm_original':float(np.linalg.norm(av)),
                           'edit_norm_fronted':float(np.linalg.norm(bv)),
                           'common_effect_minimax_over_larger_edit':float(mismatch/(2*den)) if den else None,
                           'edit_cosine':float(np.vdot(av,bv)/(np.linalg.norm(av)*np.linalg.norm(bv))) if np.linalg.norm(av)*np.linalg.norm(bv) else None}
        reports.append({'pair_id':r['pair_id'],'task_edit_geometry':metrics,
                        'common_effect_lower_bound_over_native_effect_scale':
                            {k:v/2 if v is not None else None for k,v in r['errors']['interaction'].items()}})
    values=[r['common_effect_lower_bound_over_native_effect_scale']['vocabulary'] for r in reports]
    result={'controls':controls(),'parent_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'reports':reports,'vocabulary_common_effect_bound_range':[min(values),max(values)],
            'pairs_excluding_uniform_10_percent_context_free_effect':sum(v>.1 for v in values),
            'scope':'Sharp context-free effect prediction bound on the fixed source-stage interface, normalized by larger native source effect. Removal-reference offsets cancel. This does not exclude a shared nonlinear operation that reads explicit context, nor prove semantic circuit identity.'}
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='reports'}))


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]),Path(sys.argv[2]))
