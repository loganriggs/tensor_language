"""Describe whether aligned input spaces support self terms or cross terms."""
import json,time
from pathlib import Path
import torch
P=Path(__file__).parent
torch.set_num_threads(2)
torch.set_grad_enabled(False)
start=time.monotonic()
data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
Q=torch.stack([q for pair in data['pairs'] for q in pair['Qs']])
root=torch.linalg.inv(data['inverse_root'])
rows=[]
for geometry in ('native_isotropic','calibration_shaped'):
 T=Q if geometry=='native_isotropic' else root@Q@root
 bases=[]
 for j in range(3):
  pair=T[2*j:2*j+2]
  _,U=torch.linalg.eigh((pair@pair).sum(0))
  bases.append(U[:,-352:])
 for a,b in ((0,1),(0,2),(1,2)):
  U,s,Vh=torch.linalg.svd(bases[a].T@bases[b],full_matrices=False)
  shared=torch.linalg.qr(bases[a]@U[:,:64]+bases[b]@Vh.T[:,:64],mode='reduced').Q
  for consumer in (a,b):
   pair=T[2*consumer:2*consumer+2]
   qp=pair@shared
   core=shared.T@qp
   cross=qp-shared@core
   rest=pair-shared@core@shared.T-shared@cross.transpose(-1,-2)-cross@shared.T
   energy=float(pair.square().sum())
   fractions=[float(core.square().sum())/energy,2*float(cross.square().sum())/energy,float(rest.square().sum())/energy]
   assert abs(sum(fractions)-1)<1e-10
   rows.append(dict(geometry=geometry,edge=[a,b],consumer=consumer,mean_squared_cosine=float(s[:64].square().mean()),within_shared_energy=fractions[0],shared_to_remaining_cross_energy=fractions[1],remaining_energy=fractions[2],orthogonal_energy_replay=abs(sum(fractions)-1)))
out=dict(records=rows,seconds=time.monotonic()-start,scope='Descriptive decomposition of native quadratic coefficient energy around each target-derived midpoint principal64 input subspace. Each edge is assessed separately; energies across edges overlap and must not be added. Remaining space is the entire orthogonal complement, not the private224 graph branch. No fitted graph, causal identity, cost saving or acceptance claim.')
(P/'ALIGNED_INTERACTIONS_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
