"""Exact-polynomial quotient reduction of all saved native state readout norms."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from quadratic_state_readout import execute
from quartic_two_square_quotient import reduce_gram,evaluate_norm,controls

P=Path(__file__).resolve().parent
source=P.parent/'bilinear_quotient/circuits/followups/source_ood_v2_state_readout_result.json'
r=json.loads(source.read_text());assert r['predictions']['pred_a_instrument']
torch.set_num_threads(2)
contexts=[];checks=[];replays=[];ranks=[]
for c in r['contexts']:
    compiled={}
    for arm,raw in c['compiled_cores'].items():
        numerator=np.array(raw['numerator']);triangular=np.array(raw['norm_triangular'])
        norm=[]
        for packed in triangular:
            matrix=np.zeros((3,3));matrix[np.triu_indices(3)]=packed
            coefficients,check=reduce_gram(matrix.T@matrix)
            norm.append(coefficients);checks.append(check['coefficient_error']);ranks.append(check['rank'])
        norm=np.array(norm)
        original={k:torch.tensor(v,dtype=torch.float64) for k,v in raw.items()}
        for radius in [-2.,-1.,-.5,0.,.25,.5,1.,2.]:
            n=numerator[:,0]+radius*numerator[:,1]+radius**2*numerator[:,2]
            q=evaluate_norm(norm,radius)+np.finfo(np.float32).eps
            values=(30*np.tanh(n/(30*np.sqrt(q[:,None])))).reshape(len(n),4,2)
            values=values[:,:,0]-values[:,:,1]
            replays.append(float(np.max(abs(values-execute(original,radius).numpy()))))
        compiled[arm]=dict(numerator=numerator.tolist(),norm_two_squares=norm.tolist())
    contexts.append(dict(panel=c['panel'],role=c['role'],template=c['template'],compiled_cores=compiled))
assert len(checks)==288 and max(replays)<1e-9
out=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),planted_control=controls(),
         native_programs=len(checks),max_coefficient_error=max(checks),max_readout_replay=max(replays),
         norm_features_before=3,norm_features_after=2,coefficients_before=30,coefficients_after=29,
         rank_histogram={str(k):ranks.count(k) for k in set(ranks)},contexts=contexts,
         scope='Equivalent per-ray readout programs, not new native prediction evidence. Full-six10.103%prediction failure remains; native generators and gauge nonuniqueness remain.')
(P/'NATIVE_TWO_SQUARE_NORM_V1_RESULT.json').write_text(json.dumps(out,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='contexts'},indent=2))
