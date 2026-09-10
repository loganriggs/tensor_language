"""Average six exact paths across the observed value/routing/recipient cube."""
from itertools import permutations
from pathlib import Path
import hashlib,json
import numpy as np
from mixed_state_projector_v1 import projector


def contributions(cube):
    cube=np.asarray(cube,dtype=float)
    assert cube.shape[:3]==(2,2,2)
    out=np.zeros((3,)+cube.shape[3:])
    for order in permutations(range(3)):
        vertex=[0,0,0]
        for axis in order:
            before=cube[tuple(vertex)];vertex[axis]=1
            out[axis]+=(cube[tuple(vertex)]-before)/6
    change=cube[1,1,1]-cube[0,0,0]
    error=float(np.max(abs(out.sum(0)-change)))
    return out,change,error


def main():
    p=Path(__file__).resolve().parent;source=p/'MATURE_VALUE_ROUTE_V1_RESULT.json'
    data=json.loads(source.read_text());assert data['predictions']['pred_a_instrument']
    q=projector(data['corners'],2,4);reports=[]
    for row in data['reports']:
        mixed=np.einsum('ij,vprjc->vpric',q,np.asarray(row['effects']))
        foil=1 if row['foil']=='himself' else 2
        objects={'margin':mixed[...,0]-mixed[...,foil],'readers':mixed-mixed.mean(-1,keepdims=True)}
        report={'world_id':row['world_id']}
        for name,cube in objects.items():
            parts,d,error=contributions(cube);assert error<1e-12
            den=float(np.sum(d*d));assert den>0
            report[name]={'closure_max_abs':error,'parts':{
                label:{'signed_projection':float(np.sum(part*d)/den),
                       'relative_norm':float(np.linalg.norm(part)/np.linalg.norm(d))}
                for label,part in zip(('value','routing','recipient'),parts)}}
        reports.append(report)
    # Exact additive and multiplicative controls separate diagonal attribution
    # from interaction: xyz assigns equal thirds of its diagonal change.
    additive=np.zeros((2,2,2));product=np.zeros_like(additive)
    for x in range(2):
        for y in range(2):
            for z in range(2):additive[x,y,z]=x+2*y+3*z;product[x,y,z]=x*y*z
    assert np.allclose(contributions(additive)[0],[1,2,3])
    assert np.allclose(contributions(product)[0],[1/3]*3)
    result={'reports':reports,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'controls':{'additive_path_accounting':True,'product_path_accounting':True},
            'scope':'Exact average of six specified context-change paths. Signed attribution can cancel and distributes interactions; it neither removes the measured interactions nor rescues failed one-factor predictions.'}
    with (p/'MATURE_FACTOR_PATH_ATTRIBUTION_V1_RESULT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({name:{axis:[min(r[name]['parts'][axis]['signed_projection'] for r in reports),
                                max(r[name]['parts'][axis]['signed_projection'] for r in reports)]
                          for axis in ('value','routing','recipient')} for name in ('margin','readers')},indent=2))


if __name__=='__main__':main()
