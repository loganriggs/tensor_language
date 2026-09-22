"""Learn missing quartic products under matched value and response objectives."""
import argparse,json,time
import torch
from toy_local_quartic_residual import planted
from quartic_cp_profile import normalize_factors
from correlated_gaussian_cp import metric
from quartic_cp import cp_gram
from audit_conditional_residual_accounting import P

def fit_readout(G,X):
 C=torch.zeros_like(X);loss=G.new_zeros(())
 for out in range(2):
  sl=slice(2*out,2*out+2);g=G[sl,sl];x=X[out,sl];c=torch.linalg.solve(g+1e-6*torch.eye(2,dtype=g.dtype),x).detach();C[out,sl]=c;loss+=c@g@c-2*c@x+1e-6*c.square().sum()
 return loss,C

def main(supplement=False):
 torch.set_num_threads(2);start=time.monotonic();rows=[];witnesses=[];names=['generic','paired_squares','squared_quadratics','shared_quadratic_factors','fourth_powers','cross_output_shared_quadratic'];mu=torch.tensor([.1,-.2,.3,.1],dtype=torch.float64)
 for family in ([5] if supplement else range(5)):
  _,_,_,rf,rc=planted(0 if family==5 else family)
  if family==5:
   rf=[a.clone() for a in rf];rf[0][:]=rf[0][0].clone();rf[1][:]=rf[1][0].clone();xx=torch.randn(13,4,dtype=torch.float64);full=torch.stack([xx@a.T for a in rf]).prod(0)@rc.T;shared=((xx@rf[0][0])*(xx@rf[1][0]))[:,None]*(((xx@rf[2].T)*(xx@rf[3].T))@rc.T);assert float((full-shared).norm()/full.norm())<1e-12
  rb=[f@mu for f in rf];norms={}
  for name,rho in [('value',None),('response',.5)]:norms[name]=((rc.T@rc)*metric(rf,rb,rf,rb,rho)).sum()
  norms['coefficient']=((rc.T@rc)*cp_gram(rf,rf)).sum()
  witnesses.append(dict(family=names[family],atoms_per_output=2,known_factors_exact=True,value_norm=float(norms['value']),response_norm=float(norms['response'])))
  for optimizer,rate in [('adam',.1),('muon',.01)]:
   for seed in [0,1]:
    for objective,rho in [('value',None),('response',.5)]:
     torch.manual_seed(24100+seed);params=[torch.randn(4,4,dtype=torch.float64,requires_grad=True) for _ in range(4)];opt=(torch.optim.Adam if optimizer=='adam' else torch.optim.Muon)(params,lr=rate);best=None
     for step in range(251):
      fs=normalize_factors(params);bs=[f@mu for f in fs];G=metric(fs,bs,fs,bs,rho);X=rc@metric(rf,rb,fs,bs,rho);loss,C=fit_readout(G,X);value=float(loss.detach())
      if best is None or value<best[0]:best=(value,step,[f.detach().clone() for f in fs],C.clone())
      if step==250:break
      opt.zero_grad();(loss/norms[objective]).backward();opt.step()
     value,selected,fs,C=best;bs=[f@mu for f in fs];errors={}
     with torch.no_grad():
      for name,query in [('value',None),('response',.5),('coefficient','coefficient')]:
       if query=='coefficient':G=cp_gram(fs,fs);X=rc@cp_gram(rf,fs)
       else:G=metric(fs,bs,fs,bs,query);X=rc@metric(rf,rb,fs,bs,query)
       energy=norms[name]+((C.T@C)*G).sum()-2*(C*X).sum();assert float(energy)>-1e-9*float(norms[name]);errors[name+'_error']=float((energy.clamp_min(0)/norms[name]).sqrt())
     row=dict(family=names[family],optimizer=optimizer,lr=rate,seed=seed,objective=objective,selected_step=selected,**errors);rows.append(row);print(json.dumps(row),flush=True)
 summary=[]
 for optimizer in ['adam','muon']:
  values=[r for r in rows if r['optimizer']==optimizer and r['objective']=='value'];responses=[r for r in rows if r['optimizer']==optimizer and r['objective']=='response'];count=sum(r['response_error']<.05 for r in responses);ratio=sum(r['response_error'] for r in responses)/sum(r['response_error'] for r in values);summary.append(dict(optimizer=optimizer,response_fits_below_5percent=count,mean_response_error_ratio=ratio,pred_feasibility=None if supplement else count>=8,pred_comparison=None if supplement else ratio<=.9))
 out=dict(rows=rows,summary=summary,witnesses=witnesses,steps=250,seconds=time.monotonic()-start,scope=('Descriptive8fit supplement: cross-output shared quadratic, fifth distinct structural class; no original prediction applies.' if supplement else 'Original40fits: five configurations but four polynomial structural classes; two square configurations are slot permutations. Known2atoms/output capacity, value-tuned rates, own-objective checkpoint selection; no native adoption.'))
 (P/('PAIRED_RESIDUAL_SHARING_SUPPLEMENT_V1.json' if supplement else 'PAIRED_RESIDUAL_LEARNING_TOYS_V1.json')).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--supplement',action='store_true');args=parser.parse_args();main(args.supplement)
