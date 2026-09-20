"""Read registered native cells; preserve per-condition failures and precision floor."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent
r=json.loads((p.parent/'bilinear_quotient/circuits/followups/v4_edge_amplitude_transfer_v1_result.json').read_text())
rows=[]
for arm in ['exact','carry','linear']:
 for s,a in [(1.,1.),(.5,1.),(1.,.5),(-1.,1.),(1.,-1.),(2.,1.),(1.,2.)]:
  xs=[x for x in r['records'] if x['arm']==arm and (x['subject_amplitude'],x['attractor_amplitude'])==(s,a)]
  assert len(xs)==8
  rows.append(dict(arm=arm,subject=s,attractor=a,passing=sum(x['passed'] for x in xs),max_number=max(x['errors'][0] for x in xs),max_control=max(max(x['errors'][1:]) for x in xs),min_weak_norm=min(x['weak_norm'] for x in xs),failures=[dict(panel=x['panel'],family=x['family'],errors=x['errors']) for x in xs if not x['passed']]))
result=dict(predictions=r['predictions'],rows=rows,scope='Frozen core and selectors; independently varied amplitudes on opened texts, native singleton backgrounds regenerated; not text OOD or token-input extraction')
f=p/'V4_EDGE_AMPLITUDE_TRANSFER_CPU_AUDIT.json';assert not f.exists();f.write_text(json.dumps(result,indent=2)+'\n')
for x in rows:print(x['arm'],x['subject'],x['attractor'],x['passing'],x['max_number'],x['max_control'])
print(r['predictions'])
