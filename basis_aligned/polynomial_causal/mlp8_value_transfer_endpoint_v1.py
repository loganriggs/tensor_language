"""Descriptive endpoint concentration audit; no refitting or new confirmation."""
import json
from pathlib import Path
import torch
P=Path(__file__).resolve().parent
rows=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows']
z=torch.load(P/'MLP8_VALUE_TRANSFER_V1_ARTIFACT.pt',weights_only=True)['regional']
city=[i for i,r in enumerate(rows) if r['family']==0]
def score(ix):
 assert len(ix)%2==0
 assert all(rows[ix[j]]['donor_id']==ix[j+1] for j in range(0,len(ix),2))
 t=z[ix]; c=t[::2,0,0]-t[1::2,0,0]
 directed=t[:,2,0]-t[:,0,0]; directed[::2]*=-1
 return dict(pairs=len(ix)//2, native_contrast=float(c.mean()),transfer=float(directed.mean()/c.mean()),positive=int((directed>0).sum()),directions=len(ix))
concepts=sorted({rows[i]['concept'] for i in city})
result=dict(scope='Reused city donor panel; descriptive endpoint concentration, not held-out confirmation.',aggregate=score(city),endpoints={str(c):score([i for i in city if rows[i]['concept']==c]) for c in concepts},leave_one_out={str(c):score([i for i in city if rows[i]['concept']!=c]) for c in concepts})
old=json.loads((P/'MLP8_VALUE_TRANSFER_V1_RESULT.json').read_text())['records'][0]['donor_transfer']
result['aggregate_replay_error']=abs(old-result['aggregate']['transfer'])
assert result['aggregate_replay_error']<1e-12
(P/'MLP8_VALUE_TRANSFER_ENDPOINT_V1_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
