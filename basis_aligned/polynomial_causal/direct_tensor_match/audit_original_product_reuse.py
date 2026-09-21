"""Can the exact leading-mode product dictionary serve native modes2/3?

Fixed common products, optimal linear coefficients under declared coefficient
metrics. Exact centered constant/linear branches retained for native probes.
Not a data-free or native-behavior confirmation; no graph topology optimization.
"""
from pathlib import Path
import torch,json,time
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter();program=torch.load(p/'MIDPOINT_ORIGINAL_SOURCE_BLOCK_V1.pt',weights_only=True);mode=torch.load(p/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True);cal=torch.load(p/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);rows=torch.load(p/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True)
ck='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin';state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu');L=state['transformer.h.16.mlp.Left.weight'].double();R=state['transformer.h.16.mlp.Right.weight'].double();D=state['transformer.h.16.mlp.Down.weight'].double();lam=state['transformer.h.17.lambdas'].double()[0]
z=cal['z'].flatten(0,1).double();mu=z.mean(0);delta=z-mu;cov=delta.T@delta/len(delta);i,j,kind=program['product_indices'];T=program['shared_reader'];u=T[:,i];v=T[:,j];A=torch.where(kind==1,u+v,u);B=torch.where(kind==1,u-v,v);features=(delta@A)*(delta@B);feature_means=(A*(cov@B)).sum(0)
Qa=[];labels=[]
for k in range(3):
 for side in ['A','B']:
  channel=lam*(D.T@mode[side][:,k]);raw=L.T@(channel[:,None]*R);Qa.append((raw+raw.T)/2);labels.append(f'{side}{k+1}')
truth=torch.stack([((z@Q)*z).sum(1) for Q in Qa],1);base=torch.stack([mu@Q@mu+torch.trace(cov@Q)+delta@(2*Q@mu) for Q in Qa],1)
scale=cal['recipient_scale'].flatten().double();n=rows['n'].flatten(0,1).double();m=rows['m'].flatten(0,1).double();hread=((n+.5*m)@mode['A'][:,:3])*scale[:,None];alpha=mode['mean_n']@mode['A'][:,:3];beta=mode['mean_m']@mode['B'][:,:3]
def modes(q):return ((hread-.5*q[:,0::2])/scale[:,None]-alpha)*(q[:,1::2]/scale[:,None]-beta)
def phi(q):return modes(q).sum(1)
truephi=phi(truth);records=[];exports={};control=[]
for metric in ['isotropic','covariance']:
 M=torch.eye(1152,dtype=torch.float64) if metric=='isotropic' else cov
 MA=M@A;MB=M@B;gram=.5*((A.T@MA)*(B.T@MB)+(A.T@MB)*(B.T@MA));norm=gram.diag().sqrt();G=gram/norm[:,None]/norm[None,:];eig,V=torch.linalg.eigh(G);cut=eig.max()*1e-12;active=eig>cut
 rhs=torch.stack([(A*(M@Q@MB)).sum(0) for Q in Qa],1)/norm[:,None];coeff=(V[:,active]@((V[:,active].T@rhs)/eig[active,None]))/norm[:,None]
 pred=base+(features-feature_means)@coeff
 for k,Q in enumerate(Qa):
  approximation=.5*((A*coeff[:,k])@B.T+(B*coeff[:,k])@A.T);diff=Q-approximation;num=torch.trace(M@diff@M@diff);den=torch.trace(M@Q@M@Q);coeferr=float((num.clamp_min(0)/den)**.5)
  records.append(dict(metric=metric,reader=labels[k],coefficient_relative_error=coeferr,native_source_variation_error=float((pred[:,k]-truth[:,k]).norm()/(truth[:,k]-truth[:,k].mean()).norm()),linear_only_variation_error=float((base[:,k]-truth[:,k]).norm()/(truth[:,k]-truth[:,k].mean()).norm())))
  if k<2:control.append(coeferr)
 exports[metric]=dict(mode_reference_variation_norms=[float((modes(truth)[:,j]-modes(truth)[:,j].mean()).norm()) for j in range(3)],mode_error_norms=[float((modes(pred)[:,j]-modes(truth)[:,j]).norm()) for j in range(3)],normal_equation_relative_residual=float((G@(coeff*norm[:,None])-rhs).norm()/rhs.norm()),per_mode_variation_errors=[float((modes(pred)[:,j]-modes(truth)[:,j]).norm()/(modes(truth)[:,j]-modes(truth)[:,j].mean()).norm()) for j in range(3)],coefficients=coeff,gram_smallest_eigenvalue=float(eig.min()),gram_condition=float(eig.max()/eig[active].min()),discarded_directions=int((~active).sum()),rank3_product_variation_error=float((phi(pred)-truephi).norm()/(truephi-truephi.mean()).norm()))
new=[r for r in records if r['metric']=='covariance' and r['reader'] not in ['A1','B1']];result=dict(predictions=dict(pred_a_original_pair=max(control)<1e-5,pred_b_reuse=all(r['coefficient_relative_error']<=.25 and r['native_source_variation_error']<=.10 for r in new),pred_c_rank3=exports['covariance']['rank3_product_variation_error']<=.10),records=records,metric_summaries={k:{a:b for a,b in v.items() if a!='coefficients'} for k,v in exports.items()},seconds=time.perf_counter()-start,scope='Fixed exact native-leading-pair product dictionary. Additional original mode2/3source forms projected by linear least squares under isotropic/covariance coefficient metrics; exact centered affine branches retained. Native source and rank3scalar errors only calibration probes. No fresh/nativebehavior or semanticreuse claim.')
(p/'MIDPOINT_ORIGINAL_PRODUCT_REUSE_V1.json').write_text(json.dumps(result,indent=2)+'\n');torch.save({k:v['coefficients'] for k,v in exports.items()},p/'MIDPOINT_ORIGINAL_PRODUCT_REUSE_V1.pt');print(json.dumps(result,indent=2))

# Optional separately registered follow-up; reuse the already reconstructed problem.
import sys
if '--residual-screen' in sys.argv:
 ev,V=torch.linalg.eigh(cov);root=(V*ev.clamp_min(0).sqrt())@V.T;inv=(V*ev.clamp_min(1e-30).rsqrt())@V.T;coeff=exports['covariance']['coefficients'];baseline=base+(features-feature_means)@coeff;spectra={}
 for family in ['direct','reuse_residual']:
  spectra[family]=[]
  for k,Q in enumerate(Qa[2:],2):
   Qreuse=.5*((A*coeff[:,k])@B.T+(B*coeff[:,k])@A.T);target=Q if family=='direct' else Q-Qreuse;ev,vec=torch.linalg.eigh(root@target@root);order=ev.abs().argsort(descending=True);spectra[family].append((ev[order],inv@vec[:,order]))
 follow=[]
 for family in ['direct','reuse_residual']:
  for rank in [0,4,16]:
   pred=base.clone() if family=='direct' else baseline.clone();pred[:,:2]=truth[:,:2]
   for k,(values,readers) in enumerate(spectra[family],2):
    P=readers[:,:rank];weights=values[:rank];mean=(P*(cov@P)).sum(0);pred[:,k]+=(((delta@P).square()-mean)*weights).sum(1)
   source_errors=[float((pred[:,j]-truth[:,j]).norm()/(truth[:,j]-truth[:,j].mean()).norm()) for j in range(2,6)];mode_errors=[float((modes(pred)[:,j]-modes(truth)[:,j]).norm()/(modes(truth)[:,j]-modes(truth)[:,j].mean()).norm()) for j in range(3)];follow.append(dict(family=family,private_rank_per_new_reader=rank,new_source_variation_errors=source_errors,mode_variation_errors=mode_errors,combined_variation_error=float((phi(pred)-truephi).norm()/(truephi-truephi.mean()).norm()),source_products=1152+4*rank,stored_float_scalars=1152**2+(14 if family=='reuse_residual' else 10)*1152+4*1152*rank+4*rank+10))
 chosen=next(r for r in follow if r['family']=='reuse_residual' and r['private_rank_per_new_reader']==16);control=next(r for r in follow if r['family']=='direct' and r['private_rank_per_new_reader']==16)
 result=dict(predictions=dict(pred_a_source=max(chosen['new_source_variation_errors'])<=.10,pred_b_modes=max(chosen['mode_variation_errors'][1:])<=.15,pred_c_vs_direct=all(a<=b for a,b in zip(chosen['new_source_variation_errors'],control['new_source_variation_errors']))),records=follow,scope='Calibration-only comparison using original weights and covariance. Leadingmode originalsourcepair retained exactly. Directprivate forms versus residualprivate forms at sameadditionalrank; reuse charges extra4608outputweights. Empirical scalar errors, not native causal confirmation.')
 (p/'MIDPOINT_PRODUCT_REUSE_RESIDUAL_SCREEN_V1.json').write_text(json.dumps(result,indent=2)+'\n');print('RESIDUAL SCREEN',json.dumps(result,indent=2))
