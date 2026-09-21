"""Oracle scalar repair bound on already-opened intervention responses."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent
source=json.loads((P/'PAIRED_ROOT_INTERCHANGE_V1.json').read_text())
result=[]
for s in source['summary']:
 if s['arm']=='exact':continue
 rows=[r for r in source['rows'] if all(r[k]==s[k] for k in ['domain','alpha','arm'])]
 R=sum(r['reference_energy'] for r in rows);E=sum(r['estimate_energy'] for r in rows);D=sum(r['dot'] for r in rows);err=sum(r['error_energy'] for r in rows)
 assert abs(err-(R+E-2*D))/max(R,1e-30)<1e-10
 scale=D/E;floor=math.sqrt(max(0,1-D*D/(R*E)))
 result.append(dict(domain=s['domain'],alpha=s['alpha'],arm=s['arm'],original_error=s['effect_error'],oracle_scale=scale,oracle_error_floor=floor,remaining_squared_error_fraction=floor**2/(err/R)))
out=dict(scope='Best scalar on complete opened-cell logit response, an optimistic descriptive correction, not a realizable refit through nonlinear endpoints. No independent validation.',results=result)
(P/'QUARTIC_RESPONSE_SCALE_AUDIT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
