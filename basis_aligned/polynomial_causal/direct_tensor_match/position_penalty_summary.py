from pathlib import Path
import json
p=Path(__file__).resolve().parent;r=json.loads((p/'MIDPOINT_POSITION_AUDIT_V1.json').read_text());rows=[]
for d in ['fineweb','code']:
 for f in ['removal','same_token','source_only','context_only']:
  ratios=[]
  for bin in ['0:64','64:128','128:256']:
   get=lambda name:next(v['effect_error'] for v in r['records'] if v['domain']==d and v['family']==f and v['program']==name and v['position_bin']==bin)
   ratios.append(get('separate256')/get('exact_linear'))
  rows.append(dict(domain=d,family=f,early_mid_late_ratios=ratios,late_over_early_penalty=ratios[-1]/ratios[0]))
out=p/'MIDPOINT_POSITION_PENALTY_SUMMARY_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=rows,scope='Successor CPU ratio audit of same fixed interventions. Recipient-position association, not causal sequence-length manipulation. Broad early degradation rejects an exclusively late-position account.'),indent=2)+'\n');print(out.read_text())
