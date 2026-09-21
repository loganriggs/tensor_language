from pathlib import Path
import torch,json,time
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');torch.set_num_threads(2);start=time.perf_counter();torch.set_default_dtype(torch.float64)
old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);pack=torch.load(p/'MIDPOINT_SHARED_FIT_PROGRAMS_V1.pt',weights_only=True);Sn=pack['regularized_calibration_roots']['n'];Sm=pack['regularized_calibration_roots']['m'];A=old['A'].double();B=old['B'].double();Xn,Rn=torch.linalg.qr(Sn@A,mode='reduced');Xm,Rm=torch.linalg.qr(Sm@B,mode='reduced')
def cores(a,b):return torch.einsum('igr,jgr->gij',a.reshape(16,4,4),b.reshape(16,4,4))
teacher=cores(Rn,Rm);normalizer=teacher.norm();teacher=teacher/normalizer
# Verify compact basis contraction against full weighted Frobenius error on a perturbation.
g=torch.Generator().manual_seed(261101);delta=torch.randn(16,16,generator=g);compact=delta.square().sum();expanded=(Xn@delta@Xm.T).square().sum();replay=float((compact-expanded).abs()/compact);assert replay<1e-12
base=pack['programs']['native_moment_8'];a0=Xn.T@Sn@base['Pn'].double()@base['Tn'].double()/normalizer.sqrt();b0=Xm.T@Sm@base['Pm'].double()@base['Tm'].double()/normalizer.sqrt();Ua,_,_=torch.linalg.svd(a0,full_matrices=False);Ub,_,_=torch.linalg.svd(b0,full_matrices=False);init=[Ua[:,:8],Ua[:,:8].T@a0,Ub[:,:8],Ub[:,:8].T@b0];initial=float((cores(a0,b0)-teacher).norm());fits=[];best=None
for seed in [0,1,2]:
 for lr in [.003,.01,.03]:
  gen=torch.Generator().manual_seed(261101+seed);params=[(v.clone()+(torch.randn(v.shape,generator=gen)*.005 if seed else 0)).requires_grad_() for v in init];opt=torch.optim.Adam(params,lr=lr);bestloss=float('inf');checkpoint=None
  for step in range(1200):
   opt.zero_grad();Pn,Tn,Pm,Tm=params;loss=(cores(Pn@Tn,Pm@Tm)-teacher).square().sum();loss.backward();opt.step()
   if float(loss.detach())<bestloss:bestloss=float(loss.detach());checkpoint=[v.detach().clone() for v in params]
  # Recompute after the update; do not label a pre-update loss as checkpoint loss.
  Pn,Tn,Pm,Tm=checkpoint;value=float((cores(Pn@Tn,Pm@Tm)-teacher).square().sum());fits.append(dict(seed=seed,lr=lr,steps=1200,relative_weighted_error=value**.5))
  if best is None or value<best[0]:best=(value,checkpoint,dict(seed=seed,lr=lr))
Pn,Tn,Pm,Tm=best[1];left=torch.linalg.solve(Sn,Xn)@Pn;right=torch.linalg.solve(Sm,Xm)@Pm;program=dict(Pn=left.float(),Pm=right.float(),Tn=(Tn*normalizer.sqrt()).float(),Tm=(Tm*normalizer.sqrt()).float(),offset=old['offset'],readout=old['readout'],scalar_readers=old['scalar_readers'],reduced_writers=old['reduced_writers'])
result=dict(initial_relative_weighted_error=initial,best_relative_weighted_error=best[0]**.5,best=best[2],fits=fits,compact_metric_replay=replay,seconds=time.perf_counter()-start,scope='CPU exact separable weighted matching to confirmed16product program. Only shared topology refit; no native target activations or held data. Native fidelity not yet tested.')
out=p/'MIDPOINT_JOINT_CORE_REFIT_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(program=program),p/'MIDPOINT_JOINT_CORE_REFIT_V1.pt');print(json.dumps(result,indent=2))
