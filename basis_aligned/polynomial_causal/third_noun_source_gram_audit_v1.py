"""Post-result native source geometry, not an independent causal-source test."""
import hashlib,json
from pathlib import Path
import numpy as np


def geometry(gram, selected):
    total = float(gram.sum())
    if total <= 0: raise ValueError('live source sum required')
    projection = float(gram[selected].sum()/total)
    square = float(gram[np.ix_(selected, selected)].sum()/total)
    return {'signed_projection': projection, 'norm_ratio': float(np.sqrt(max(0.,square))),
            'source_sum_relative_error': float(np.sqrt(max(0.,1+square-2*projection)))}


def main():
    p=Path(__file__).resolve().parent;source=p/'THIRD_NOUN_MIXED_STATE_V1_RESULT.json'
    data=json.loads(source.read_text());assert all(data['predictions'].values())
    keys=data['source_order'];grams=np.array(data['source_gram']);reports=[]
    groups={kind:[i for i,k in enumerate(keys) if k.startswith(kind)] for kind in ('attn','mlp')}
    for start in (0,6,12):groups[f'layers_{start}_{start+5}']=[i for i,k in enumerate(keys) if start<=int(k.split(':')[1])<start+6]
    for w,g in enumerate(grams):
        assert np.allclose(g,g.T,atol=1e-8,rtol=1e-12)
        eig=np.linalg.eigvalsh(g);assert eig[0]>=-1e-10*max(1.,eig[-1])
        total=float(g.sum());native=data['final_mixed_norm_squared_by_world'][w]
        closure=abs(total-native)/native;assert closure<1e-5
        per_source={k:geometry(g,[i]) for i,k in enumerate(keys)}
        assert abs(sum(x['signed_projection'] for x in per_source.values())-1)<1e-10
        reports.append({'world_id':data['reports'][w]['world_id'],
            'native_squared_norm_closure_relative':closure,
            'sum_source_norms_over_sum_norm':float(np.sqrt(np.maximum(0,np.diag(g))).sum()/np.sqrt(total)),
            'groups':{k:geometry(g,v) for k,v in groups.items()},'sources':per_source,
            'top_signed_sources':sorted(keys,key=lambda k:per_source[k]['signed_projection'],reverse=True)[:5]})
    # Positive and cancellation controls distinguish norm from signed contribution.
    vectors=np.array([[3.,0.],[-2.,0.],[0.,1.]])
    g=vectors@vectors.T
    assert geometry(g,[0])['signed_projection']==1.5
    assert geometry(g,[1])['signed_projection']==-1.
    assert geometry(g,[2])['signed_projection']==.5
    assert geometry(g,[0,1,2])['source_sum_relative_error']==0
    result={'passed':True,'model_forwards':0,'reports':reports,'controls_passed':True,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Exact geometry of transported native writes. Correlated/contextual producers retained; these are not independent causal removals or learned shared circuit units.'}
    with (p/'THIRD_NOUN_SOURCE_GRAM_AUDIT_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k!='sources'} for r in reports],indent=2))


if __name__=='__main__':main()
