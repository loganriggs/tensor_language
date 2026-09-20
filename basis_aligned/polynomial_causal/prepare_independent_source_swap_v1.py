"""Remove the six-source warm-start dependency from the fixed five-port selector."""
import json
import hashlib
from pathlib import Path
import numpy as np
from quadratic_budgeted_direction import choose
from budgeted_modal_direction import choose as linear_choose

P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/six_source_complement_v1r1_result.json'
r=json.loads(source.read_text());keep=[0,2,3,4,5];B=np.array([0.,1.,1.,1.,0.])
contexts=[];cells=[]
for c in r['contexts']:
    g=np.array(c['gradient'])[:,:,keep];h=np.array(c['hessian'])[:,:,keep][:,:,:,keep]
    selected=[];checks=[]
    for gi,hi in zip(g,h):
        null,_=linear_choose(gi,0.,reference=B)
        ai,check=choose(gi,hi,null,reference=B)
        selected.append(ai);checks.append(check)
    a=np.array(selected);expanded=np.zeros((len(a),6));expanded[:,keep]=a
    pred=-np.einsum('boi,bi->bo',g,a)-.5*np.einsum('bi,boij,bj->bo',a,h,a)
    base=-np.einsum('boi,i->bo',g,B)-.5*np.einsum('i,boij,j->bo',B,h,B)
    pattern='SEMANTIC_PORT_FRESH' if c['dataset']=='opened' else 'SOURCE_OOD_V1'
    rows=json.loads((P/f"{pattern}_{c['panel'].upper()}_ROWS.json").read_text())
    rows=[x for x in rows if x['template']==c['template']]
    contexts.append(dict(dataset=c['dataset'],panel=c['panel'],role=c['role'],template=c['template'],
                         omitted_source=1,amplitudes=expanded.tolist(),checks=checks))
    for family in dict.fromkeys(x['family'] for x in rows):
        ids=[i for i,x in enumerate(rows) if x['family']==family];y,yb=pred[ids],base[ids]
        cells.append(dict(dataset=c['dataset'],panel=c['panel'],role=c['role'],family=family,
                          retention=float(y[:,0]@yb[:,0]/max(yb[:,0]@yb[:,0],1e-30)),
                          modal_ratio=float(max(np.linalg.norm(y[:,1:],axis=0))/max(np.linalg.norm(y[:,0]),1e-30))))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),contexts=contexts,cells=cells,
         predicted_joint_passes=sum(c['retention']>=.8 and c['modal_ratio']<=.1 for c in cells),
         scope='Only fixed five-source g/H supplied to selector. Native validation pending; source generation still uses native prefix.')
(P/'INDEPENDENT_SOURCE_SWAP_V1.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['contexts','cells']},indent=2))
