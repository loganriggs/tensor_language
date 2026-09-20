#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_metric_replay pred_b_weighted_gain pred_c_heldout_gain
"""Native metric study: 16 fits, width128/512 x 4 metrics x 2 seeds, Muon .005 600steps.
Predictions: a float64 noncentral quadrature error<1e-10 and coordinate replay<2e-4;
b best width512 noncentral Gaussian training error<.5;
c best width512 noncentral heldout empirical error below best width512 isotropic.
Null: covariance weighting does not improve empirical heldout matching. Price3*1152*width
plus fixed QR/R frames; no native forwards, no CE or circuit identification claim.
"""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(widths=[128,512],metrics=['isotropic','centered_covariance','second_moment','noncentral'],seeds=[0,1],steps=600,lr=.005,ridge_fraction=.001,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P));from noncentral_quadratic import gaussian_inner
 from implicit_quadratic import inner
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'NATIVE_METRIC_SWEEP_V1.json';assert not out.exists();start=time.perf_counter()
 capture=torch.load(P/'NATIVE_COVARIANCE_V1.pt',weights_only=False);panels=capture['panels'];cal=panels[0];re=capture['R'].cuda().float()
 snap='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin';state=torch.load(snap,map_location='cpu',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  qu,ru=torch.linalg.qr(w('lm_head.weight'));c0=ru@w('transformer.h.17.mlp.Down.weight');a0=w('transformer.h.17.mlp.Left.weight')@re.T;b0=w('transformer.h.17.mlp.Right.weight')@re.T
  rows=[p['rows'].cuda().float() for p in panels];truth=[((q@a0.T)*(q@b0.T))@c0.T for q in rows]
  isoenergy=inner(c0,a0,b0,c0,a0,b0);frobenergy=inner(c0,a0,b0,c0,a0,b0,False)
 records=[];best={};transforms={};maxreplay=0.
 for metric in PLAN['metrics']:
  with torch.no_grad():
   if metric=='isotropic':chol=torch.eye(1152,device='cuda');mean=None;ridge=0.
   else:
    cov=cal['second_moment'] if metric=='second_moment' else cal['covariance'];ridge=PLAN['ridge_fraction']*float(cov.trace())/1152
    chol=torch.linalg.cholesky(cov.cuda()+ridge*torch.eye(1152,device='cuda',dtype=torch.float64)).float()
    mean=torch.linalg.solve_triangular(chol,cal['mean'].cuda().float()[:,None],upper=False).flatten() if metric=='noncentral' else None
   a,b=a0@chol,b0@chol;scale=gaussian_inner(c0,a,b,c0,a,b,mean).sqrt();c=c0/scale
   teacher=float(gaussian_inner(c,a,b,c,a,b,mean));transforms[metric]=dict(ridge=ridge,scale=float(scale))
  for width in PLAN['widths']:
   for seed in PLAN['seeds']:
    torch.manual_seed(seed)
    # Pair identical physical functions across metrics, before dividing by each teacher norm.
    dd=torch.randn(1152,width,device='cuda')*.1/math.sqrt(1152*width)*isoenergy.sqrt()/scale
    ll=torch.randn(width,1152,device='cuda');rr=torch.randn(width,1152,device='cuda');ll=ll/ll.norm(dim=1,keepdim=True);rr=rr/rr.norm(dim=1,keepdim=True)
    ll=ll@chol;rr=rr@chol;ln=ll.norm(dim=1);rn=rr.norm(dim=1)
    d=torch.nn.Parameter(dd*ln*rn);l=torch.nn.Parameter(ll/ln[:,None]);r=torch.nn.Parameter(rr/rn[:,None])
    opt=torch.optim.Muon([d,l,r],lr=PLAN['lr'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');history=[];valbest=float('inf');saved=None
    for step in range(PLAN['steps']):
     opt.zero_grad();ll=l/l.norm(dim=1,keepdim=True);rr=r/r.norm(dim=1,keepdim=True)
     loss=(teacher+gaussian_inner(d,ll,rr,d,ll,rr,mean)-2*gaussian_inner(c,a,b,d,ll,rr,mean))/teacher
     assert bool(torch.isfinite(loss));value=float(loss.detach())
     if value<valbest:valbest=value;saved=[v.detach().clone() for v in (d,ll,rr)]
     if step%100==0 or step==PLAN['steps']-1:history.append(dict(step=step,loss=value))
     loss.backward();opt.step()
     for group in opt.param_groups:group['lr']=PLAN['lr']*(.01+.99*.5*(1+math.cos(math.pi*(step+1)/PLAN['steps'])))
    with torch.no_grad():
     dd,ll,rr=saved;dd=dd*scale
     aq=torch.linalg.solve_triangular(chol.T,ll.T,upper=True).T;bq=torch.linalg.solve_triangular(chol.T,rr.T,upper=True).T
     pred=[((q@aq.T)*(q@bq.T))@dd.T for q in rows]
     empirical=[float((v-t).norm()/t.norm()) for v,t in zip(pred,truth)]
     whiteq=torch.linalg.solve_triangular(chol,rows[0].T,upper=False).T
     replay=float((((whiteq@ll.T)*(whiteq@rr.T))@dd.T-pred[0]).norm()/pred[0].norm());maxreplay=max(maxreplay,replay)
     ge=float((isoenergy+inner(dd,aq,bq,dd,aq,bq)-2*inner(c0,a0,b0,dd,aq,bq))/isoenergy)
     fe=float((frobenergy+inner(dd,aq,bq,dd,aq,bq,False)-2*inner(c0,a0,b0,dd,aq,bq,False))/frobenergy)
    row=dict(metric=metric,width=width,seed=seed,training_relative_error=math.sqrt(max(0,valbest)),isotropic_relative_error=math.sqrt(max(0,ge)),frobenius_relative_error=math.sqrt(max(0,fe)),empirical_calibration_error=empirical[0],empirical_evaluation_error=empirical[1],coordinate_replay=replay,parameter_values=3*1152*width,history=history);records.append(row);print(json.dumps(row),flush=True)
    key=f'{metric}_{width}'
    if key not in best or valbest<best[key]['loss']:best[key]=dict(loss=valbest,seed=seed,factors=[v.cpu() for v in (dd,aq,bq)])
    result=dict(plan=PLAN,records=records,transforms=transforms,seconds=time.perf_counter()-start,scope='Gaussian moment objectives from native calibration statistics, whitened-coordinate optimization, paired physical random initialization; evaluation rows never optimized. Empirical fourth moments used only for diagnostics. No causal adoption.')
    out.write_text(json.dumps(result,indent=2)+'\n')
 # Select restart by training objective, not heldout performance.
 selected=[min([r for r in records if r['metric']==m and r['width']==512],key=lambda r:r['training_relative_error']) for m in ['isotropic','noncentral']]
 check=json.load(open(P/'NONCENTRAL_CHECK_V1.json'))
 result['predictions']=dict(pred_a_metric_replay=maxreplay<2e-4 and check['quadrature_absolute_error']<1e-10,pred_b_weighted_gain=selected[1]['training_relative_error']<.5,pred_c_heldout_gain=selected[1]['empirical_evaluation_error']<selected[0]['empirical_evaluation_error'])
 guard_torch_save(dict(best=best,R=capture['R'],scope='Factors in q coordinates; output QR frame reproducible from checkpoint.'),str(P/'NATIVE_METRIC_SWEEP_V1.pt'));out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
