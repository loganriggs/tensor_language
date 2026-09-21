"""Registered CPU screen, native six-read algebra unchanged across geometries."""
from pathlib import Path
import json,time,torch
from quadratic_center_screen import screen
P=Path(__file__).parent
torch.set_num_threads(2);torch.set_grad_enabled(False)
start=time.perf_counter()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
Q=torch.stack([q for pair in d['pairs'] for q in pair['Qs']])
energy=Q.square().sum((-1,-2)).reshape(3,2).sum(1)
T=Q/energy.repeat_interleave(2).sqrt()[:,None,None]
S=torch.linalg.inv(d['inverse_root']);rows=[]
for geometry,forms in [('native_isotropic',T),('covariance_shaped',S@T@S)]:
 for seed in [23016,23017]:
  g=torch.Generator().manual_seed(seed);a=torch.randn(6,dtype=T.dtype,generator=g);b=torch.randn(6,dtype=T.dtype,generator=g)
  result=screen(forms,a,b);rows.append(dict(geometry=geometry,seed=seed,screen=result));print(geometry,seed,result,flush=True)
out=dict(seconds=time.perf_counter()-start,records=rows,pred_a='PASS' if all(r['screen']['instrument']=='PASS' for r in rows) else 'INCONCLUSIVE',pred_b='PASS' if all(r['screen'].get('primary_components',1)>1 for r in rows) else 'FAIL',pred_c='PASS' if all(rows[j]['screen'].get('primary_sizes')==rows[j+2]['screen'].get('primary_sizes') for j in range(2)) else 'FAIL',scope='Exact independent-coordinate block hypothesis only; fixed target, two congruent input geometries, two fixed pencils. Numerical screen is not an interval certificate or lower bound on approximation by overlapping arithmetic DAGs.')
(P/'NATIVE_QUADRATIC_CENTER_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='records'},indent=2))
