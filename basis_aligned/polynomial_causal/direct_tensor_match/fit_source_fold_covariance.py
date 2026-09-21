"""Isotropic versus covariance-weighted upstream quadratic extraction baselines.

Keep exact constant/linear Taylor terms and empirical quadratic mean. Fit only
centered quadratic coefficient structure by eigendecomposition; no paired-output
regression. pred_a replay exact full form<1e-10; pred_b weighted rank16 source
variation and downstream product errors<=.10; pred_c weighted rank16 product
error below isotropic rank16. Calibration screen, not native/fresh adoption.
"""
from pathlib import Path
import json,torch,time
p=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.perf_counter()
out=p/'MIDPOINT_SOURCE_COVARIANCE_FIT_V1.json';assert not out.exists()
assert all(json.loads((p/'MIDPOINT_SOURCE_FOLD_CAPTURE_V1.json').read_text())['predictions'].values())
data=torch.load(p/'MIDPOINT_SOURCE_FOLD_CALIBRATION_V1.pt',weights_only=True);fold=torch.load(p/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.pt',weights_only=True);mode=torch.load(p/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True)
z=data['z'].flatten(0,1).double();mu=z.mean(0);delta=z-mu;cov=delta.T@delta/len(delta);ev,V=torch.linalg.eigh(cov);mask=ev>1e-10*ev.max();root=(V*(ev.clamp_min(0)*mask).sqrt())@V.T;inv=(V*torch.where(mask,ev.clamp_min(1e-30).rsqrt(),0))@V.T
scale=data['recipient_scale'].flatten(0,1).double()[:,0];hread=data['h_reader'].flatten().double();alpha=mode['mean_n']@mode['A'][:,0];beta=mode['mean_m']@mode['B'][:,0]
true=torch.stack([((z@fold[k]['matrix'])*z).sum(1) for k in ['a','b']],1)
product=lambda q:((hread-.5*q[:,0])/scale-alpha)*(q[:,1]/scale-beta)
phi=product(true);sources={};checks=[]
for k in ['a','b']:
 Q=fold[k]['matrix'];linear=2*Q@mu;constant=mu@Q@mu+torch.trace(cov@Q);reconstructed=constant+delta@linear+((delta@Q)*delta).sum(1)-torch.trace(cov@Q)
 checks.append(float((reconstructed-true[:,['a','b'].index(k)]).norm()/true[:,['a','b'].index(k)].norm()))
 spectra={}
 for metric in ['isotropic','covariance']:
  target=Q if metric=='isotropic' else root@Q@root;values,vectors=torch.linalg.eigh(target);order=values.abs().argsort(descending=True);values=values[order];vectors=vectors[:,order];readers=vectors if metric=='isotropic' else inv@vectors
  spectra[metric]=dict(values=values,readers=readers)
 sources[k]=dict(linear=linear,constant=constant,spectra=spectra)
records=[];exports={}
for metric in ['isotropic','covariance']:
 for rank in [0,1,4,16,64,256]:
  approximations=[];e=dict(mu=mu,alpha=alpha,beta=beta,a=mode['A'][:,0],writer=mode['writer'],R_U=mode['R_U'])
  for k in ['a','b']:
   s=sources[k];P=s['spectra'][metric]['readers'][:,:rank];values=s['spectra'][metric]['values'][:rank];center=(P*(cov@P)).sum(0)
   pred=s['constant']+delta@s['linear']+(((delta@P).square()-center)*values).sum(1)
   approximations.append(pred);e[k+'_reader']=P.clone();e[k+'_eigenvalues']=values.clone();e[k+'_quadratic_mean']=center;e[k+'_linear']=s['linear'];e[k+'_constant']=s['constant']
  pred=torch.stack(approximations,1);ph=product(pred);variation=true-true.mean(0)
  row=dict(metric=metric,rank_per_source=rank,source_variation_errors=[float((pred[:,j]-true[:,j]).norm()/variation[:,j].norm()) for j in [0,1]],product_relative_error=float((ph-phi).norm()/phi.norm()),product_variation_error=float((ph-phi).norm()/(phi-phi.mean()).norm()),source_weight_coefficients=2*1152*(rank+1)+2*rank,source_shared_mean_entries=1152,source_other_constants=2*rank+2)
  records.append(row);exports[f'{metric}_{rank}']=e
weighted=next(r for r in records if r['metric']=='covariance' and r['rank_per_source']==16);isotropic=next(r for r in records if r['metric']=='isotropic' and r['rank_per_source']==16)
result=dict(predictions=dict(pred_a_instrument=max(checks)<1e-10,pred_b_small_extraction=max(weighted['source_variation_errors']+[weighted['product_variation_error']])<=.10,pred_c_metric_help=weighted['product_variation_error']<isotropic['product_variation_error']),records=records,exact_expansion_replay=max(checks),covariance_discarded_trace_fraction=float(ev.clamp_min(0)[~mask].sum()/ev.clamp_min(0).sum()),seconds=time.perf_counter()-start,scope='Original upstream quadratic weights. Exact mean and linear terms retained; mean-corrected quadratic truncation. Weighted coefficient norm corresponds to centered Gaussian quadratic variance up to factor2, not a claim of Gaussian native inputs. Empirical calibration errors measured without target regression; no fresh/native behavior validation. h_reader and recipient scale remain native input ports; source-only price excludes final writer and h read.')
torch.save(exports,p/'MIDPOINT_SOURCE_COVARIANCE_PROGRAMS_V1.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(out.read_text())
