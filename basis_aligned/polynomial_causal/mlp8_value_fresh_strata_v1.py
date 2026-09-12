"""Decompose context failure by city pair and endpoint, without refitting."""
from pathlib import Path
import json,torch
P=Path(__file__).resolve().parent
rows=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows']
z=torch.load(P/'MLP8_VALUE_FRESH_V1_ARTIFACT.pt',weights_only=True)['regional'][24:]
def score(ix):
 assert all(rows[ix[j]]['donor_id']==ix[j+1] for j in range(0,len(ix),2))
 t=z[ix];c=t[::2,0,0]-t[1::2,0,0];e=t[:,2,0]-t[:,0,0];e[::2]*=-1
 return dict(native=float(c.mean()),donor=float(e.mean()/c.mean()),donor_positive=int((e>0).sum()),directions=len(ix))
cells=[]
for family in range(3):
 base=[i for i,r in enumerate(rows) if r['family']==family]
 cells.append(dict(family=family,aggregate=score(base),city_pairs=[dict(variant=v,score=score([i for i in base if rows[i]['variant']==v])) for v in range(2)],endpoints=[dict(concept=c,score=score([i for i in base if rows[i]['concept']==c])) for c in range(6)]))
old=json.loads((P/'MLP8_VALUE_FRESH_V1_RESULT.json').read_text())['records']
error=max(abs(c['aggregate']['donor']-r['donor_transfer']) for c,r in zip(cells,old));assert error<1e-12
out=dict(replay_error=error,cells=cells,scope='Reused fresh panel; descriptive construction/citypair/endpoint audit, no row exclusion or threshold revision.')
(P/'MLP8_VALUE_FRESH_STRATA_V1_DIAGNOSTIC.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps([dict(family=c['family'],city_pairs=c['city_pairs'],endpoint_donors=[r['score']['donor'] for r in c['endpoints']]) for c in cells],indent=2))
