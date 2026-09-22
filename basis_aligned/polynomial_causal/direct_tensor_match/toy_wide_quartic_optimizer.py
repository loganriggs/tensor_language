"""Known-capacity quartic fits at small and native ambient input widths."""
import argparse,json,time
import torch
from toy_local_quartic_residual import planted
from toy_paired_residual_learning import fit_readout
from quartic_cp_profile import normalize_factors
from correlated_gaussian_cp import metric
from quartic_cp import cp_gram
from audit_conditional_residual_accounting import P

def main(unit_initial=False):
 torch.set_num_threads(2);start=time.monotonic();rows=[];witnesses=[]
 for family in [0,1,3,4,5]:
  _,_,_,rf0,rc=planted(0 if family==5 else family)
  rf0=[a.clone() for a in rf0]
  if family==5:rf0[0][:]=rf0[0][0].clone();rf0[1][:]=rf0[1][0].clone()
  for d in ([1152] if unit_initial else [4,1152]):
   rf=[torch.nn.functional.pad(a,(0,d-4)) for a in rf0];mu=torch.zeros(d,dtype=torch.float64);mu[:4]=torch.tensor([.1,-.2,.3,.1]);rb=[f@mu for f in rf]
   norms={name:((rc.T@rc)*metric(rf,rb,rf,rb,rho)).sum() for name,rho in [('value',None),('response',.5)]};norms['coefficient']=((rc.T@rc)*cp_gram(rf,rf)).sum()
   small=metric(rf0,[f@mu[:4] for f in rf0],rf0,[f@mu[:4] for f in rf0]);witness=float((metric(rf,rb,rf,rb)-small).abs().max());assert witness<1e-10;witnesses.append(dict(family=family,d=d,padded_gram_replay=witness))
   for seed in [0,1]:
    g=torch.Generator().manual_seed(24100+seed);base=[torch.randn(4,4,dtype=torch.float64,generator=g) for _ in range(4)]
    extra=torch.Generator().manual_seed(40000+seed);initial=[torch.cat([a,torch.randn(4,d-4,dtype=torch.float64,generator=extra)],dim=1) for a in base]
    before=normalize_factors(initial)
    if unit_initial:
     initial=[a.clone() for a in before];assert max(float((a-b).abs().max()) for a,b in zip(normalize_factors(initial),before))<1e-14
    for optimizer in (['muon_original'] if unit_initial else ['adam','muon_original','muon_match_rms']):
     params=[a.clone().requires_grad_() for a in initial];rate=.1 if optimizer=='adam' else .01
     opt=torch.optim.Adam(params,lr=rate) if optimizer=='adam' else torch.optim.Muon(params,lr=rate,adjust_lr_fn=None if optimizer=='muon_original' else 'match_rms_adamw');best=None
     for step in range(251):
      fs=normalize_factors(params);bs=[f@mu for f in fs];G=metric(fs,bs,fs,bs);X=rc@metric(rf,rb,fs,bs);loss,C=fit_readout(G,X);value=float(loss.detach())
      if best is None or value<best[0]:best=(value,step,[f.detach().clone() for f in fs],C.detach().clone())
      if step==250:break
      opt.zero_grad();(loss/norms['value']).backward();assert all(torch.isfinite(a.grad).all() for a in params);opt.step()
     with torch.no_grad():
      _,step,fs,C=best;bs=[f@mu for f in fs];errors={}
      for name,rho in [('value',None),('response',.5),('coefficient','coefficient')]:
       G=cp_gram(fs,fs) if name=='coefficient' else metric(fs,bs,fs,bs,rho);X=rc@(cp_gram(rf,fs) if name=='coefficient' else metric(rf,rb,fs,bs,rho));energy=norms[name]+((C.T@C)*G).sum()-2*(C*X).sum();assert energy>-1e-9*norms[name];errors[name+'_error']=float((energy.clamp_min(0)/norms[name]).sqrt())
      init=normalize_factors(initial);angles=torch.cat([torch.rad2deg(torch.acos((a*b).sum(1).clamp(-1,1))) for a,b in zip(init,fs)]);nuisance=torch.cat([f[:,4:].square().sum(1) for f in fs])
     row=dict(unit_initial=unit_initial,family=family,d=d,seed=seed,optimizer=optimizer,lr=rate,selected_step=step,median_rotation_degrees=float(angles.median()),mean_nuisance_factor_energy=float(nuisance.mean()),**errors);rows.append(row);print(json.dumps(row),flush=True)
     (P/('WIDE_QUARTIC_UNIT_INITIAL_PARTIAL_V1.json' if unit_initial else 'WIDE_QUARTIC_OPTIMIZER_PARTIAL_V1.json')).write_text(json.dumps(dict(rows=rows,scope='Running partial receipt, not final result'),indent=2)+'\n')
 wide={o:[r['value_error'] for r in rows if r['d']==1152 and r['optimizer']==o] for o in ['adam','muon_original','muon_match_rms']};median=lambda a:float(torch.tensor(a,dtype=torch.float64).quantile(.5));pred=dict(unit_initial_recovery=sum(e<.05 for e in wide['muon_original'])>=8) if unit_initial else dict(adam_recovery=sum(e<.05 for e in wide['adam'])>=8,muon_limited=sum(e<.05 for e in wide['muon_original'])<=2,muon_scaled_gain=median(wide['muon_match_rms'])<=.8*median(wide['muon_original']))
 (P/('WIDE_QUARTIC_UNIT_INITIAL_V1.json' if unit_initial else 'WIDE_QUARTIC_OPTIMIZER_V1.json')).write_text(json.dumps(dict(rows=rows,witnesses=witnesses,predictions=pred,seconds=time.monotonic()-start,scope=('Unit raw-parameter initialization control; ten wide default-Muon fits, initial normalized function unchanged; native untested.' if unit_initial else 'Five distinct planted quartic structures, two restarts, two ambient dimensions, three fixed configurations; exact Gaussian errors, native remains untested.')),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--unit-initial',action='store_true');args=parser.parse_args();main(args.unit_initial)
