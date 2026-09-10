"""Signed observed route accounting and exact frozen-write carry control."""
import json,hashlib
from pathlib import Path
import numpy as np
from mixed_state_projector_v1 import projector
from mlp8_consumer_effect_geometry_v1 import factorial_effects


def carry_control():
    rng=np.random.default_rng(910807);x=rng.normal(size=(32,16));delta=rng.normal(size=x.shape)*.1
    injection=rng.normal(size=(4,32,16));writes=rng.normal(size=(4,32,16))
    scales=np.array([.7,-1.1,.8,1.2]);biases=np.array([.2,.3,-.1,.4])
    def recurrence(state):
        for a,b,e,w in zip(scales,biases,injection,writes):state=a*state+b*e+w
        return state
    native=recurrence(x);edited=recurrence(x-delta);pred=native-np.prod(scales)*delta
    weights=rng.normal(size=(3,16))
    def read(state):
        norm=state/np.sqrt((state**2).mean(-1,keepdims=True)+np.finfo(np.float32).eps)
        return 30*np.tanh((norm@weights.T)/30)
    errors={'state':float(np.max(abs(pred-edited))), 'nonlinear_reader':float(np.max(abs(read(pred)-read(edited))))}
    assert max(errors.values())<1e-12
    # A live nonlinear write breaks the frozen-write condition.
    native_live=native+native**2;edited_live=edited+edited**2
    wrong=native_live-np.prod(scales)*delta
    failure=float(np.linalg.norm(read(wrong)-read(edited_live)));assert failure>.01
    return {'passed':True,'errors':errors,'live_nonlinear_write_falsifier_error':failure,
            'scope':'Product of residual multipliers is exact only with later writes held fixed; native reader normalization and softcap may remain nonlinear.'}


def main():
    p=Path(__file__).resolve().parent;f=p/'MLP8_ATTENTION9_FACTORIAL_V1_RESULT.json';data=json.loads(f.read_text());assert data['predictions']['pred_a_instrument']
    q=projector(data['corners'],2,4);reports=[]
    for row in data['reports']:
        parts=factorial_effects(row['cube_logits']);foil=1 if row['foil']=='himself' else 2
        effects={k:q@v for k,v in zip(('total','attention','bypass','interaction'),parts)}
        report={'world_id':row['world_id'],'layout':row['layout']}
        for readout in ('margin','readers'):
            e={k:v[:,0]-v[:,foil] if readout=='margin' else v-v.mean(-1,keepdims=True) for k,v in effects.items()}
            total=e['total'];den=float(np.sum(total*total));assert den>0
            report[readout]={k:{'signed_projection':float(np.sum(v*total)/den),'relative_norm':float(np.linalg.norm(v)/np.sqrt(den))} for k,v in e.items() if k!='total'}
            assert abs(sum(v['signed_projection'] for v in report[readout].values())-1)<1e-12
        reports.append(report)
    result={'reports':reports,'carry_control':carry_control(),'source_sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
            'scope':'Native-background route attribution; direct-carry control is synthetic, not a native dominance finding.'}
    with (p/'SOURCE_ATTENTION_ROUTE_ACCOUNTING_V1_RESULT.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    for layout in ('original','fronted_pp'):
        rows=[r for r in reports if r['layout']==layout]
        print(layout,{k:[min(r['margin'][k]['signed_projection'] for r in rows),max(r['margin'][k]['signed_projection'] for r in rows)] for k in ('attention','bypass','interaction')})
    print(result['carry_control'])

if __name__=='__main__':main()
