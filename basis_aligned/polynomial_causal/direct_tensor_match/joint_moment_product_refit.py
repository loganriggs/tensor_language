from pathlib import Path
import torch,json,time,sys
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');sys.path.insert(0,str(p));from joint_product_search import tensor
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);start=time.perf_counter();old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);pack=torch.load(p/'MIDPOINT_SHARED_FIT_PROGRAMS_V1.pt',weights_only=True);Sn=pack['regularized_calibration_roots']['n'];Sm=pack['regularized_calibration_roots']['m'];Qn,Rn=torch.linalg.qr(Sn@old['A'].double(),mode='reduced');Qm,Rm=torch.linalg.qr(Sm@old['B'].double(),mode='reduced');T=tensor((old['readout'].T.double(),Rn,Rm));scale=T.norm();T=T/scale;An=torch.linalg.solve(Sn,Qn);Bm=torch.linalg.solve(Sm,Qm);rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();nr=n@An;mr=m@Bm;phi=torch.einsum('bi,bj->bij',nr,mr).flatten(1);moment=phi.T@phi/len(phi);targetnorm=(T.flatten(1)*(T.flatten(1)@moment)).sum().detach();warm=torch.load(p/'MIDPOINT_JOINT_PRODUCTS_V1.pt',weights_only=True)['programs'][8];init=[warm['readout'].T.double(),Qn.T@Sn@warm['A'].double()/scale.sqrt(),Qm.T@Sm@warm['B'].double()/scale.sqrt()]
def loss(factors):
 delta=(tensor(factors)-T).flatten(1);return (delta*(delta@moment)).sum()/targetnorm
# Exact loss and gradient contraction against all paired sample evaluations.
test=[v.clone().requires_grad_() for v in init];implicit=loss(test);delta=(tensor(test)-T).flatten(1);direct=(phi@delta.T).square().sum()/len(phi)/targetnorm;g1=torch.autograd.grad(implicit,test,retain_graph=True);g2=torch.autograd.grad(direct,test);replay=float((implicit-direct).abs().detach()/direct.detach());grad=max(float((a-b).norm()/b.norm()) for a,b in zip(g1,g2));assert replay<1e-10 and grad<1e-10
fits=[];best=None;initial=float(loss(init))**.5
for seed in [0,1,2]:
 for lr in [.003,.01]:
  gen=torch.Generator().manual_seed(261118+seed);params=[(v.clone()+(torch.randn(v.shape,generator=gen)*.001 if seed else 0)).requires_grad_() for v in init];opt=torch.optim.Adam(params,lr=lr);value=float('inf');checkpoint=None
  for step in range(1200):
   opt.zero_grad();objective=loss(params);current=float(objective.detach())
   if current<value:value=current;checkpoint=[v.detach().clone() for v in params]
   objective.backward();opt.step()
  fits.append(dict(seed=seed,lr=lr,steps=1200,relative_joint_error=value**.5))
  if best is None or value<best[0]:best=(value,checkpoint,dict(seed=seed,lr=lr))
w,a,b=best[1];A=An@a*scale.sqrt();B=Bm@b*scale.sqrt();program=dict(A=A.float(),B=B.float(),readout=w.T.float(),offset=old['offset'],scalar_readers=old['scalar_readers'],reduced_writers=old['reduced_writers']);pred=((n@A)*(m@B))@w.T-old['offset'].double();truth=rows['y'].flatten(0,1).double()@old['scalar_readers']-old['offset'].double();confirmed=((n@old['A'].double())*(m@old['B'].double()))@old['readout'].double()-old['offset'].double();result=dict(initial_joint_error=initial,best_joint_error=best[0]**.5,calibration_error_to_native=float((pred-truth).norm()/truth.norm()),calibration_error_to_confirmed=float((pred-confirmed).norm()/confirmed.norm()),weighted_coefficient_error=float((tensor(best[1])-T).norm()),metric_replay=replay,gradient_replay=grad,best=best[2],fits=fits,seconds=time.perf_counter()-start,scope='Exact empirical joint lifted moment on originalcalibration inputs. Weight-derived confirmedprogram target, rank8products. No heldfitting, no native swap claim; objective denominator is target rawsecondmoment, not centeredvariation.')
out=p/'MIDPOINT_JOINT_MOMENT_REFIT_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(program=program),p/'MIDPOINT_JOINT_MOMENT_REFIT_V1.pt');print(json.dumps(result,indent=2))
