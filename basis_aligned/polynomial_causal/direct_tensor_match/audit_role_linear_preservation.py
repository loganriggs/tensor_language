"""Broad preservation audit after a source-specific gate passes."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent;r=json.loads((p/'MIDPOINT_ROLE_LINEAR_NATIVE_V1.json').read_text());records=[]
for d,s in r['summary'].items():
 for f in ['removal','same_token','source_only','context_only']:
  baseline=s['product_exact_linear'][f]['centered_effect_relative_error'];candidate=s['product_separate256'][f]['centered_effect_relative_error']
  records.append(dict(domain=d,family=f,baseline_error=baseline,candidate_error=candidate,ratio=candidate/baseline,within_five_percent=candidate<=1.05*baseline))
result=dict(records=records,all_families_preserved=all(v['within_five_percent'] for v in records),scope='Successor audit of all existing intervention families; not a retroactive change to registered source gate. Same reused panels. Explicitly distinguishes source-only preservation from full-function preservation.')
out=p/'MIDPOINT_ROLE_LINEAR_PRESERVATION_AUDIT_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
