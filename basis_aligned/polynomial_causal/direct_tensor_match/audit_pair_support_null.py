"""Cross-pair input-space compatibility against independent relative orientations."""
from pathlib import Path
import json,time,torch
P=Path(__file__).parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic();plan=json.loads((P/'PAIR_SUPPORT_NULL_PLAN_V1.json').read_text())
data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);Q=torch.stack([q for pair in data['pairs'] for q in pair['Qs']]);S=torch.linalg.inv(data['inverse_root']);d=Q.shape[-1];width=plan['pair_width'];rank=plan['edge_width'];records=[]
def score(bases):
 values=[float(torch.linalg.svdvals(bases[a].T@bases[b])[:rank].square().mean()) for a,b in ((0,1),(0,2),(1,2))]
 return sum(values)/3,values
rng=torch.Generator().manual_seed(plan['seed_base']-1);O=torch.linalg.qr(torch.randn(d,d,dtype=Q.dtype,generator=rng)).Q
null=[]
for seed in range(plan['null_samples']):
 rng=torch.Generator().manual_seed(plan['seed_base']+seed);frames=[torch.linalg.qr(torch.randn(d,width,dtype=Q.dtype,generator=rng),mode='reduced').Q for _ in range(3)];value,edges=score(frames);null.append(dict(seed=plan['seed_base']+seed,statistic=value,edge_values=edges))
for geometry in plan['metrics']:
 A=S if geometry=='calibration_shaped' else torch.eye(d,dtype=Q.dtype);T=A@Q@A;bases=[]
 for j in range(3):
  pair=T[2*j:2*j+2];_,U=torch.linalg.eigh((pair@pair).sum(0));bases.append(U[:,-width:])
 value,edges=score(bases);rotated,_=score([O@b for b in bases]);replay=abs(value-rotated);assert replay<1e-10
 vals=torch.tensor([r['statistic'] for r in null],dtype=Q.dtype);records.append(dict(geometry=geometry,native_statistic=value,edge_values=edges,common_rotation_replay=replay,null_mean=float(vals.mean()),null_std=float(vals.std()),null_min=float(vals.min()),null_max=float(vals.max()),exceeds_all_nulls=value>float(vals.max()),monte_carlo_upper_tail=(1+sum(r['statistic']>=value for r in null))/(len(null)+1)))
result=dict(plan=plan,records=records,null_samples=null,predictions=dict(pred_a=True,pred_b=all(r['exceeds_all_nulls'] for r in records)),seconds=time.monotonic()-start,scope='Input supports derived independently from pair mode-Gram spectra. Same null frames used in both geometries because width and ambient dimension match. Not architecture-preserving, semantic or causal evidence.')
(P/'PAIR_SUPPORT_NULL_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(records,indent=2))
