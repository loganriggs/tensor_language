"""Posthoc cost and intervention comparison; no candidate fitting."""
from pathlib import Path
import json
p=Path(__file__).resolve().parent;uniform=json.loads((p/'MIDPOINT_CENTERED_ALLOCATION_V1.json').read_text());adaptive=json.loads((p/'MIDPOINT_ADAPTIVE_ALLOCATION_NATIVE_V1.json').read_text());rows=[]
for d in ['fineweb','code']:
 for family in ['removal','same_token','source_only','context_only']:
  a=adaptive['summary'][d]['product_adaptive_mixed512'][family]['centered_effect_relative_error'];u=uniform['summary'][d]['product_w256r4'][family]['centered_effect_relative_error']
  rows.append(dict(domain=d,family=family,adaptive_error=a,uniform1024_error=u,ratio=a/u))
result=dict(records=rows,products_ratio=.5,weight_ratio=4423680/5308416,maximum_effect_error_ratio=max(v['ratio'] for v in rows),scope='Exploratory comparison of mixed-rank512 versus uniform256x4 with same exact first-order terms. Not preregistered gate, not comparison at equal coefficient storage to old compact512. Reused panels, no fresh confirmation.')
out=p/'MIDPOINT_ALLOCATION_TRADEOFF_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
