"""Cross-panel coefficient reuse and a joint-feasibility redteam of failures."""
import json,hashlib
from pathlib import Path
import numpy as np
from shared_selective_source_lp import choose
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/refined_selective_sources_v1_result.json'
a=json.loads(source.read_text());reference=np.array([0.,0.,1.,1.,1.,0.])[a['parent_mapping']];records=[];compatibility=[]
for held in a['gradients']:
    if held['panel']!='congruent':continue
    role,template=held['role'],held['template'];g=np.asarray(held['gradient']);families=held['families']
    donor=next(v for v in a['amplitudes'] if v['panel']=='opposite' and v['role']==role and v['template']==template)
    fitg=np.asarray(next(v for v in a['gradients'] if v['panel']=='opposite' and v['role']==role and v['template']==template)['gradient'])
    donor_a=np.asarray(donor['amplitudes']['oracle']);held_a=np.asarray(next(v for v in a['amplitudes'] if v['panel']=='congruent' and v['role']==role and v['template']==template)['amplitudes']['oracle'])
    for mode,weights in [('same_lexical_pair',donor_a),('next_noun_same_number',np.roll(donor_a,-2,axis=0)),('held_oracle',held_a)]:
        pred=-np.einsum('boi,bi->bo',g,weights);baseline=-g[:,0]@reference
        for family in dict.fromkeys(families):
            ids=[i for i,f in enumerate(families) if f==family];n=pred[ids,0];b=baseline[ids]
            retention=float(n@b/(b@b));leak=float(np.linalg.norm(pred[ids,1:],axis=0).max()/max(np.linalg.norm(n),1e-30))
            records.append(dict(role=role,template=template,family=family,mode=mode,retention=retention,control_ratio=leak,passed=bool(retention>=.8 and leak<=.1)))
    for i in range(len(g)):
        _,check=choose(np.stack([fitg[i],g[i]]),reference)
        compatibility.append(dict(role=role,template=template,row=i,retention_ceiling=check['retention'],compatible=check['retention']>=.8,duality_gap=check['duality_gap']))
summary={mode:dict(passed=sum(r['passed'] for r in records if r['mode']==mode),count=sum(r['mode']==mode for r in records)) for mode in ['same_lexical_pair','next_noun_same_number','held_oracle']}
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),summary=summary,pairwise_compatible=sum(r['compatible'] for r in compatibility),pairwise_count=len(compatibility),records=records,compatibility=compatibility,scope='Derivative-only transfer on opened panels. Pairwise joint fit uses held derivatives and is a feasibility diagnostic, not held-out prediction. A transfer miss can reflect nonunique LP choices; native validation remains necessary.')
(P/'REFINED_COEFFICIENT_TRANSFER_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['records','compatibility']},indent=2))
