"""Weight-only support alternatives; dense-core diagnostic, not a priced circuit."""
from pathlib import Path
import json,time,torch
from local_shared_reader_graph import expand
P=Path(__file__).parent
torch.set_num_threads(2)
torch.set_grad_enabled(False)
t0=time.monotonic()
d=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)
programs=torch.load(P/'JOINT_OVERLAP_PROGRAMS_V1.pt',weights_only=True)
meta=json.loads((P/'JOINT_OVERLAP_V1.json').read_text())
S=torch.linalg.inv(d['inverse_root']);I=torch.eye(S.shape[0],dtype=S.dtype)
rows=[]
for key in ['calibration_shaped_0','native_isotropic_26301']:
 transform=S if key.startswith('calibration') else I
 prog=programs[key]
 targets=[transform@torch.stack(pair['Qs'])@transform for pair in d['pairs']]
 global_moment=sum((T@T).sum(0)/T.square().sum() for T in targets)
 _,V=torch.linalg.eigh(global_moment);common=V[:,-128:]
 bundle=expand(prog)
 for j in range(3):
  T=transform@torch.stack(d['pairs'][j]['Qs'])@transform
  moment=(T@T).sum(0)
  # E contains target outputs on the shared span. Private/common cross terms
  # appear twice in symmetric coefficient energy; the extra E E^T emphasizes them.
  E=(T@common).permute(1,0,2).reshape(len(I),-1)
  candidates={}
  for name,M in [('mode_gram',moment),('cross_emphasis',moment+E@E.T)]:
   residual=M-common@(common.T@M)-(M@common)@common.T+common@(common.T@M@common)@common.T
   _,V=torch.linalg.eigh((residual+residual.T)/2)
   U=torch.linalg.qr(torch.cat([common,V[:,-224:]],1),mode='reduced').Q
   candidates[name]=U
  _,V=torch.linalg.eigh(moment);candidates['free_pair_352']=V[:,-352:]
  U,s,_=torch.linalg.svd(transform@bundle[str(j)]['shared_reader'],full_matrices=False)
  candidates['trained_pair_span']=U[:,:int((s>1e-10*s[0]).sum())]
  for name,U in candidates.items():
   core=U.T@T@U;hat=U@core@U.T;res=T-hat
   # Orthogonal projection: residual is orthogonal to every core in this span.
   orth=float((U.T@res@U).norm()/T.norm());assert orth<1e-10
   contained=float((common-U@(U.T@common)).norm()/common.norm())
   if name in ('mode_gram','cross_emphasis'):assert contained<1e-10
   rows.append(dict(key=key,pair=j+1,method=name,rank=U.shape[1],coefficient_error=float(res.norm()/T.norm()),projection_orthogonality=orth,common_span_residual=contained))
aggregates=[]
for key in ['calibration_shaped_0','native_isotropic_26301']:
 for name in ('mode_gram','cross_emphasis','free_pair_352','trained_pair_span'):
  subset=[r for r in rows if r['key']==key and r['method']==name]
  aggregates.append(dict(key=key,method=name,error=(sum(r['coefficient_error']**2 for r in subset)/3)**.5))
out=dict(records=rows,aggregates=aggregates,seconds=time.monotonic()-t0,scope='Common128 chosen directly from equal-pair-weighted sum of target mode Grams. Dense-core diagnostic only. Shared128/private224 supports; free-pair comparator has no sharing constraint. Mode-Gram heuristics are not globally optimal. No core sparsity, cost advantage, native execution, OOD or circuit identity claim.')
(P/'WEIGHT_DERIVED_COMMON_SUPPORT_V1.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
