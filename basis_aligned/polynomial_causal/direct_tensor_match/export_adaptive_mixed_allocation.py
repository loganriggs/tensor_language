"""Adaptive budget over all already-exported factors (4 per first256, 2 per rest)."""
from pathlib import Path
import json,torch
from adaptive_output_rank_budget import allocate
from midpoint_program import product_source_delta
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False)
source=torch.load(p/'MIDPOINT_CENTERED_ALLOCATION_GRAPHS_V1.pt',weights_only=True);a4=source['w256r4'];a2=source['w512r2'];rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();n-=n.mean(0);m-=m.mean(0)
Mn=n.T@n/len(n);Mm=m.T@m/len(m)
for M in [Mn,Mm]:M+=1e-6*M.trace()/1152*torch.eye(1152,dtype=torch.float64)
A=torch.cat([a4['A'],a2['A'][:,512:]],1);B=torch.cat([a4['B'],a2['B'][:,512:]],1)
energy_flat=(A*(Mn@A)).sum(0)*(B*(Mm@B)).sum(0);energy=torch.zeros(512,4,dtype=torch.float64);energy[:256]=energy_flat[:1024].reshape(256,4);energy[256:,:2]=energy_flat[1024:].reshape(256,2)
assert float((energy[:,:-1]-energy[:,1:]).min())>-1e-6
ranks,retained=allocate(energy,512);selected=[];groups=[]
for j,r in enumerate(ranks.tolist()):
 base=4*j if j<256 else 1024+2*(j-256)
 for k in range(r):selected.append(base+k);groups.append(j)
ids=torch.tensor(selected);groups=torch.tensor(groups)
leftmeans=torch.cat([a4['base_left_mean'],a2['base_left_mean'][512:]]);rightmeans=torch.cat([a4['base_right_mean'],a2['base_right_mean'][512:]])
e={k:a2[k] for k in ['linear_n','linear_m','full_mean']};e.update(A=A[:,ids],B=B[:,ids],base_left_mean=leftmeans[ids],base_right_mean=rightmeans[ids],product_mean=torch.zeros(512,dtype=torch.float64),reduced_writers=a2['group_writers'][:,groups])
programs=torch.load(p/'MIDPOINT_ADAPTIVE_ALLOCATION_GRAPHS_V1.pt',weights_only=True);programs['adaptive_mixed512']=e
out=p/'MIDPOINT_ADAPTIVE_MIXED_GRAPHS_V1.pt';assert not out.exists();torch.save(programs,out)
old=json.loads((p/'MIDPOINT_ADAPTIVE_ALLOCATION_V1.json').read_text());result=dict(retained_energy=retained,first2_adaptive_energy=old['retained_energy'],relative_retained_energy_gain=retained/old['retained_energy']-1,active_outputs=int((ranks>0).sum()),rank_histogram={str(k):int((ranks==k).sum()) for k in range(5)},ranks=ranks.tolist(),weight_coefficients=4423680,scope='Optimal fixed512 product allocation within existing 4/2 factor pool, same regularized centered metric as original SVD. No additional decomposition or heldout fitting. Pool omits3rd/4th factors beyond output256, so not globally optimal over all512 slices at rank4.')
(p/'MIDPOINT_ADAPTIVE_MIXED_V1.json').write_text(json.dumps(result,indent=2)+'\n');print({k:v for k,v in result.items() if k!='ranks'})
