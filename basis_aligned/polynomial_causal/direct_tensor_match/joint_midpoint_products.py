from pathlib import Path
import torch,json,time,sys
from scipy.optimize import linear_sum_assignment
p=Path('/workspace/tensor_language/basis_aligned/polynomial_causal/direct_tensor_match');sys.path.insert(0,str(p));from joint_product_search import fit,tensor
torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.set_grad_enabled(False);start=time.perf_counter();choice=json.loads((p/'JOINT_PRODUCT_TOYS_V1.json').read_text())['selected_optimizer'];old=torch.load(p/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);pack=torch.load(p/'MIDPOINT_SHARED_FIT_PROGRAMS_V1.pt',weights_only=True);Sn=pack['regularized_calibration_roots']['n'];Sm=pack['regularized_calibration_roots']['m'];Qn,Rn=torch.linalg.qr(Sn@old['A'].double(),mode='reduced');Qm,Rm=torch.linalg.qr(Sm@old['B'].double(),mode='reduced');W=old['readout'].T.double();target=tensor((W,Rn,Rm));scale=target.norm();target=target/scale;oracle=float((tensor((W,Rn/scale.sqrt(),Rm/scale.sqrt()))-target).norm());assert oracle<1e-12
results=[];programs={};allfits={};stability={};rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();y=rows['y'].flatten(0,1).double()@old['scalar_readers']-old['offset'].double();baseline=((n@old['A'].double())*(m@old['B'].double()))@old['readout'].double()-old['offset'].double();den=y.square().sum();baseerror=float(((baseline-y).square().sum()/den).sqrt())
for rank in [4,8,12,16]:
 fits=[]
 for seed in [0,1,2]:
  for lr in [.003,.01]:
   with torch.enable_grad():r=fit(target,rank,choice,lr,261115+seed,1200)
   fits.append(dict(rank=rank,seed=seed,lr=lr,weighted_error=r['error'],cancellation_ratio=r['cancellation_ratio'],factors=r['factors']))
 best=min(fits,key=lambda z:z['weighted_error']);w,a,b=best['factors'];A=torch.linalg.solve(Sn,Qn)@a*scale.sqrt();B=torch.linalg.solve(Sm,Qm)@b*scale.sqrt();pred=((n@A)*(m@B))@w.T-old['offset'].double();error=float(((pred-y).square().sum()/den).sqrt());programs[rank]=dict(A=A.float(),B=B.float(),readout=w.T.float(),offset=old['offset'],scalar_readers=old['scalar_readers'],reduced_writers=old['reduced_writers']);results.append(dict(rank=rank,best_seed=best['seed'],best_lr=best['lr'],weighted_error=best['weighted_error'],native_calibration_error=error,confirmed_calibration_error=baseerror,reader_coefficients=2*1152*rank,output_mix_coefficients=4*rank,cancellation_ratio=best['cancellation_ratio']));allfits[rank]=fits
 byseed=[min([z for z in fits if z['seed']==seed],key=lambda z:z['weighted_error']) for seed in [0,1,2]];comparisons=[]
 for j in [1,2]:
  corr=torch.ones(rank,rank)
  for a,b in zip(byseed[0]['factors'],byseed[j]['factors']):corr*=((a/a.norm(dim=0)).T@(b/b.norm(dim=0)))
  ii,jj=linear_sum_assignment(-corr.numpy());values=corr[ii,jj];comparisons.append(dict(seeds=[0,j],minimum_component_tensor_cosine=float(values.min()),mean_component_tensor_cosine=float(values.mean())))
 stability[rank]=comparisons;print(results[-1],flush=True)
result=dict(optimizer=choice,oracle_rank16_error=oracle,results=results,restart_alignment=stability,fits=[{k:v for k,v in r.items() if k!='factors'} for fits in allfits.values() for r in fits],seconds=time.perf_counter()-start,scope='Joint CP products approximate confirmed4feature program in exact16dimensional weighted teacher span. Native calibration error measured only, no heldselection. Restart alignment compares signed rank-one tensor terms, invariant to internal factor rescaling. No semanticuniqueness claim.')
out=p/'MIDPOINT_JOINT_PRODUCTS_V1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(programs=programs),p/'MIDPOINT_JOINT_PRODUCTS_V1.pt');print(json.dumps(result['restart_alignment'],indent=2))
