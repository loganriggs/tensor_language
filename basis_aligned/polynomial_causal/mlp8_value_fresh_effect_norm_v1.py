"""Audit native-normalized versus branch-normalized approximation errors."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
rows=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows']
z=torch.load(P/'MLP8_VALUE_FRESH_FULL_V1_ARTIFACT.pt',weights_only=True)['regional']
def effects(t,arm,donor):
 e=t[:,arm,0]-t[:,0,0]
 if donor:e=e.clone();e[::2]*=-1;return e
 return e[1::2]-e[::2]
records=[]
for family in range(3):
 ix=[i for i,r in enumerate(rows) if r['family']==family];t=z[ix];c=t[::2,0,0]-t[1::2,0,0];rec=dict(family=family)
 for name,a,b,donor in [('removal',1,2,False),('donor',3,4,True)]:
  e4,ef=effects(t,a,donor),effects(t,b,donor);den=c.repeat_interleave(2) if donor else c
  rec[name]=dict(error_over_native=float((e4-ef).norm()/den.norm()),error_over_full_branch=float((e4-ef).norm()/ef.norm()),full_branch_norm=float(ef.norm()),four_branch_norm=float(e4.norm()),cosine=float(torch.nn.functional.cosine_similarity(e4,ef,dim=0)))
 records.append(rec)
near=[]
for concept in range(6):
 ix=[i for i,r in enumerate(rows) if r['family']==1 and r['concept']==concept];t=z[ix];c=t[::2,0,0]-t[1::2,0,0];e=effects(t,4,True)
 near.append(dict(concept=concept,fullquad_donor=float(e.mean()/c.mean()),positive=int((e>0).sum()),directions=len(ix)))
old=json.loads((P/'MLP8_VALUE_FRESH_FULL_V1_RESULT.json').read_text())['records'];err=max(abs(r[k]['error_over_native']-o[k+'_error_over_native']) for r,o in zip(records,old) for k in ['removal','donor']);assert err<1e-12
result=dict(records=records,nearquote_endpoints=near,replay_error=err,scope='Descriptive post-result error-normalization audit; original 5%native bars unchanged, no promotion or absence-of-structure claim.')
(P/'MLP8_VALUE_FRESH_EFFECT_NORM_V1_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
