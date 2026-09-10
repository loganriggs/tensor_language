"""Signed MLP8 new/inherited effect accounting; no source promotion from size."""
import hashlib,json
from pathlib import Path
import numpy as np
from mixed_state_projector_v1 import projector


def main():
    p=Path(__file__).resolve().parent;source=p/'MATURE_VALUE_MLP8_ORIGIN_V1_RESULT.json'
    data=json.loads(source.read_text());assert data['predictions']['pred_a_instrument']
    q=projector(data['corners'],2,4);reports=[]
    for row in data['reports']:
        z0=np.asarray(row['native_logits']);foil=1 if row['foil']=='himself' else 2
        effects={k:q@(z0-np.asarray(z)) for k,z in row['arm_logits'].items()}
        effects['interaction']=effects['full']-effects['new']-effects['inherited']
        report={'world_id':row['world_id'],'layout':row['layout']}
        for kind in ('margin','readers'):
            e={k:(v[:,0]-v[:,foil] if kind=='margin' else v-v.mean(-1,keepdims=True)) for k,v in effects.items()}
            full=e['full'];den=float(np.sum(full*full));assert den>0
            parts={label:{'signed_projection':float(np.sum(e[k]*full)/den),
                          'relative_norm':float(np.linalg.norm(e[k])/np.linalg.norm(full))}
                   for label,k in [('new','new'),('inherited','inherited'),('interaction','interaction')]}
            assert abs(sum(v['signed_projection'] for v in parts.values())-1)<1e-12
            report[kind]=parts
        reports.append(report)
    result={'reports':reports,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Opened-output signed attribution relative to the MLP8 partial-source effect. Negative/greater-than-one projections indicate cancellation. Neither magnitude nor projection overrides failed fidelity.'}
    with (p/'MLP8_VALUE_BRANCH_ACCOUNTING_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for layout in ('original','fronted_pp'):
        selected=[r for r in reports if r['layout']==layout]
        print(layout,{k:{v:[min(r['margin'][k][v] for r in selected),max(r['margin'][k][v] for r in selected)]
                         for v in ('signed_projection','relative_norm')} for k in ('new','inherited','interaction')})


if __name__=='__main__':main()
