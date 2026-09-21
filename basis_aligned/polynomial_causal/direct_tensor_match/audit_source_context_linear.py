"""Compare identical interface interventions before/after final nonlinearities."""
from pathlib import Path
import json,math
p=Path(__file__).resolve().parent
r=json.loads((p/'MIDPOINT_SOURCE_CONTEXT_NATIVE_RAW_V2.json').read_text())['records'];records=[]
for d in ['fineweb','code']:
 for f in ['source_only','context_only']:
  for name in ['baseline','rank8','rank32']:
   rr=[v for v in r if v['domain']==d and v['family']==f and v['candidate']=='product_'+name]
   E=sum(v['native_centered_effect_energy'] for v in rr);B=sum(v['predicted_centered_effect_energy'] for v in rr);dot=sum(v['centered_effect_dot'] for v in rr)
   records.append(dict(domain=d,family=f,program=name,linear_error=math.sqrt(sum(v['linear_error_energy'] for v in rr)/sum(v['linear_reference_energy'] for v in rr)),native_error=math.sqrt(sum(v['centered_effect_error_energy'] for v in rr)/E),native_cosine=dot/math.sqrt(E*B),predicted_norm_ratio=math.sqrt(B/E)))
result=dict(records=records,scope='Same sites/donors and full vocabulary-centered norms before versus after native final RMS/softcap. Native cosine and amplitude are descriptive pooled measures, not semantic alignment.')
out=p/'MIDPOINT_SOURCE_CONTEXT_LINEAR_AUDIT_V1.json'
if out.exists():
 assert json.loads(out.read_text())==result
 print('Existing audit reproduced exactly.')
else:
 out.write_text(json.dumps(result,indent=2)+'\n')
