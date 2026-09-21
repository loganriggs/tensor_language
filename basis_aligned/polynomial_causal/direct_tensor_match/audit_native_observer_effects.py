"""Absolute/relative native observer error audit; preserve conditional gate failure."""
from pathlib import Path
import json,torch
p=Path(__file__).resolve().parent;torch.set_num_threads(2);r=json.loads((p/'MIDPOINT_NATIVE_OBSERVER_EFFECTS_V1.json').read_text());records=[]
for candidate in ['frozen','rank1','rank2','rank3','rank8']:
 for cohort in ['all','continuation','spaced_word']:
  rr=[v for v in r['records'] if v['candidate']==candidate and v['cohort']==cohort];sites=sum(v['sites'] for v in rr);ref=sum(v['reference_energy'] for v in rr);err=sum(v['error_energy'] for v in rr)
  records.append(dict(candidate=candidate,cohort=cohort,sites=sites,native_centered_logit_rms=(ref/(sites*50304))**.5,error_centered_logit_rms=(err/(sites*50304))**.5,relative_error=(err/ref)**.5))
out=p/'MIDPOINT_NATIVE_OBSERVER_EFFECT_AUDIT_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,scope='Centered logit RMS per vocabulary coordinate and evaluated site; no threshold change or reclassification of failed spaced-word fidelity gate. CE and logit reconstruction remain distinct.'),indent=2)+'\n');print(out.read_text())
