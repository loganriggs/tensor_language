"""Opened-data five-of-six source screen; common omission, derivative-only fits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from quadratic_budgeted_direction import choose

P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/six_source_complement_v1r1_result.json'
r=json.loads(source.read_text());assert r['predictions']['pred_a_instrument']
contexts=[];cells=[]
for c in r['contexts']:
    g,h,a6=[np.array(c[k]) for k in ['gradient','hessian','amplitudes']]
    pattern='SEMANTIC_PORT_FRESH' if c['dataset']=='opened' else 'SOURCE_OOD_V1'
    rows=json.loads((P/f"{pattern}_{c['panel'].upper()}_ROWS.json").read_text())
    rows=[x for x in rows if x['template']==c['template']]
    B=np.broadcast_to([0.,0.,1.,1.,1.,0.],a6.shape)
    base=-np.einsum('boi,bi->bo',g,B)-.5*np.einsum('bi,boij,bj->bo',B,h,B)
    for omit in range(6):
        keep=[i for i in range(6) if i!=omit];a=np.zeros_like(a6);checks=[]
        for row in range(len(g)):
            gi=g[row][:,keep];hi=h[row][:,keep][:,:,keep]
            sign=np.sign(-g[row,0]@B[row]) or 1.
            reference=-sign*gi[0]/max(abs(gi[0]).max(),1e-30)
            ai,check=choose(gi,hi,a6[row,keep],reference=reference)
            a[row,keep]=ai;checks.append(check)
        pred=-np.einsum('boi,bi->bo',g,a)-.5*np.einsum('bi,boij,bj->bo',a,h,a)
        contexts.append(dict(dataset=c['dataset'],panel=c['panel'],role=c['role'],template=c['template'],
                             omitted_source=omit,amplitudes=a.tolist(),checks=checks))
        for family in dict.fromkeys(x['family'] for x in rows):
            ids=[i for i,x in enumerate(rows) if x['family']==family]
            y,yb=pred[ids],base[ids]
            cells.append(dict(dataset=c['dataset'],panel=c['panel'],role=c['role'],family=family,
                              omitted_source=omit,retention=float(y[:,0]@yb[:,0]/max(yb[:,0]@yb[:,0],1e-30)),
                              modal_ratio=float(max(np.linalg.norm(y[:,1:],axis=0))/max(np.linalg.norm(y[:,0]),1e-30))))
summary={}
for omit in range(6):
    subset=[c for c in cells if c['omitted_source']==omit]
    summary[omit]=dict(joint_passes=sum(c['retention']>=.8 and c['modal_ratio']<=.1 for c in subset),
                       min_retention=min(c['retention'] for c in subset),
                       median_retention=float(np.median([c['retention'] for c in subset])),
                       max_modal_ratio=max(c['modal_ratio'] for c in subset))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),contexts=contexts,cells=cells,summary=summary,
         scope='Common source omission; five inputs and80coefficients per context, native generators excluded. Opened-data derivative-only fit, native validation pending.')
(P/'MATCHED_WIDTH_SOURCE_SWAP_V1.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
print(json.dumps(summary,indent=2))
