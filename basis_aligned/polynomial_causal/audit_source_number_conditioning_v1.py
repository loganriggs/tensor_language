"""Distinguish source-number changes from failure to reuse coefficients."""
import json,hashlib
from pathlib import Path
import numpy as np
from shared_selective_source_lp import choose
P=Path(__file__).resolve().parent;source=P.parent/'bilinear_quotient/circuits/followups/refined_selective_sources_v1_result.json'
a=json.loads(source.read_text());reference=np.array([0.,0.,1.,1.,1.,0.])[a['parent_mapping']]
def plural(group,family):
    subject=family.endswith('|plural')
    return subject if group['role']=='subject' or group['panel']=='congruent' else not subject
fits={};groups=[];records=[]
for role in ['subject','attractor']:
    for number in [False,True]:
        fit=np.stack([np.asarray(g['gradient'])[i] for g in a['gradients'] if g['role']==role and g['panel']=='opposite' for i,f in enumerate(g['families']) if plural(g,f)==number])
        weights,check=choose(fit,reference);fits[role,str(number)]=weights
        groups.append(dict(role=role,source_plural=number,weights=weights.tolist(),fit_certificate=check,count=len(fit)))
for held in a['gradients']:
    if held['panel']!='congruent':continue
    role,template=held['role'],held['template'];g=np.asarray(held['gradient']);families=held['families'];n=len(g)
    donor=np.array(next(v for v in a['amplitudes'] if v['panel']=='opposite' and v['role']==role and v['template']==template)['amplitudes']['oracle'])
    donor_indices=np.arange(n) if role=='subject' else np.arange(n)^1
    assert all(plural(held,families[i])==plural(dict(role=role,panel='opposite'),families[j]) for i,j in enumerate(donor_indices))
    arms=dict(source_number_matched=donor[donor_indices],next_noun_source_number_matched=donor[(donor_indices+2)%n],shared_number_conditioned=np.stack([fits[role,str(plural(held,f))] for f in families]))
    for mode,weights in arms.items():
        pred=-np.einsum('boi,bi->bo',g,weights);baseline=-g[:,0]@reference
        for family in dict.fromkeys(families):
            ids=[i for i,f in enumerate(families) if f==family];x=pred[ids,0];b=baseline[ids]
            retention=float(x@b/(b@b));collateral=float(np.linalg.norm(pred[ids,1:],axis=0).max()/max(np.linalg.norm(x),1e-30))
            records.append(dict(role=role,family=family,mode=mode,retention=retention,control_ratio=collateral,passed=bool(retention>=.8 and collateral<=.1)))
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),groups=groups,records=records,summary={mode:sum(r['passed'] for r in records if r['mode']==mode) for mode in arms},scope='Derivative-only opened-data diagnostic. Source-number-conditioned shared rule has92coefficients and needs a number gate. Matching source number changes subject number in attractor transfer; neither comparison holds all context fixed. Native validation pending.')
(P/'SOURCE_NUMBER_CONDITIONING_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(summary=out['summary'],fits=[{k:v for k,v in g.items() if k!='weights'} for g in groups]),indent=2))
