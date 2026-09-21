"""Pythagorean error budget, conditional on each measured fixed output space."""
from pathlib import Path
import json,math
p=Path(__file__).resolve().parent;r=json.loads((p/'MIDPOINT_CENTERED_OUTPUT_BASIS_V1.json').read_text());target=.30;records=[]
for v in r['records']:
 if v['family']!='context_only':continue
 floor=v['linear_error'];feasible=floor<=target
 records.append(dict(domain=v['domain'],basis=v['basis'],projection_floor=floor,target_total_error=target,possible_in_span=feasible,max_inside_error_over_full_target=math.sqrt(target*target-floor*floor) if feasible else None,max_inside_error_over_projected_target=math.sqrt((target*target-floor*floor)/(1-floor*floor)) if feasible else None))
out=p/'MIDPOINT_CENTERED_OUTPUT_ERROR_BUDGET_V1.json';assert not out.exists();out.write_text(json.dumps(dict(records=records,scope='Planning bound from exact orthogonal projection in linear vocabulary-centered metric on reused diagnostics.30% is an explicit planning target, not a newly passed gate. No bound on post-nonlinearity error or alternative output spaces.'),indent=2)+'\n');print(out.read_text())
