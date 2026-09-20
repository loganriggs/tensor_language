"""Separate marginal error from true interaction error on opened joint edits."""
import hashlib
import json
from pathlib import Path
import numpy as np
from two_site_composition_scoring import key

P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/two_site_composition_v1r1_result.json'
r=json.loads(source.read_text());assert r['predictions']['pred_a_instrument']
binding=json.loads((P/'TWO_SITE_COMPOSITION_V1R1_BINDING.json').read_text())
budgets={(c['panel'],c['family']):c for c in binding['budgets']}
records=[];parities=[];closures=[]
for group in r['groups']:
    y={k:np.array(v) for k,v in group['target'].items()}
    p={k:np.array(v) for k,v in group['predictions']['state_program'].items()}
    interaction_errors={}
    for s,t in group['coordinates']:
        if not s or not t:continue
        k,ks,kt=key((s,t)),key((s,0)),key((0,t))
        interaction=y[k]-y[ks]-y[kt]
        error=(p[k]-p[ks]-p[kt])-interaction;axis=p[kt]-y[kt]
        total=(p[k]-p[ks])-(y[k]-y[ks])
        closures.append(float(np.max(abs(total-axis-error))))
        interaction_errors[k]=error
        for family in dict.fromkeys(group['families']):
            ids=[i for i,x in enumerate(group['families']) if x==family]
            den=abs(t)*budgets[group['panel'],family]['attractor_number_norm']
            norm=lambda x:(np.linalg.norm(x[ids],axis=0)/den).tolist()
            records.append(dict(panel=group['panel'],family=family,coordinate=[s,t],
                                axis_error=norm(axis),interaction_error=norm(error),
                                native_interaction_ratio=norm(interaction),total_error=norm(total)))
    for family in dict.fromkeys(group['families']):
        ids=[i for i,x in enumerate(group['families']) if x==family]
        den=budgets[group['panel'],family]['attractor_number_norm']
        for radius in [1.,.5]:
            for label,ps,pt in [('even_even',0,0),('even_odd',0,1),('odd_even',1,0),('odd_odd',1,1)]:
                component=sum(s**ps*t**pt*interaction_errors[key((radius*s,radius*t))]
                              for s in [-1,1] for t in [-1,1])/4
                parities.append(dict(panel=group['panel'],family=family,radius=radius,parity=label,
                                     error_norm_ratios=(np.linalg.norm(component[ids],axis=0)/den).tolist()))
failed=lambda value:value[0]>.1 or max(value[1:])>.05
summary={name:dict(failures=sum(failed(row[name]) for row in records),
                   number_max=max(row[name][0] for row in records))
         for name in ['axis_error','interaction_error','total_error']}
assert max(closures)<1e-12
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),summary=summary,
         max_error_partition_closure=max(closures),records=records,parities=parities,
         scope='Opened-data attribution. Parity isolates sign classes, not exact polynomial degrees; degree interpretations are leading-order hypotheses.')
(P/'TWO_SITE_FAILURE_DECOMPOSITION_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print(json.dumps([x for x in parities if x['panel']=='congruent' and x['family']=='beside_subject|plural' and x['radius']==1.],indent=2))
