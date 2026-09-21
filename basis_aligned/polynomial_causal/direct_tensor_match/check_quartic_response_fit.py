"""Five-family response-objective pilot; independent fitting and evaluation probes."""
import json,time,math
from pathlib import Path
import torch
from quartic_finite_response import fit,response_features,controls
from empirical_quartic_dictionary import features,evaluate
from shared_quadratic_bank import normalize_bank

def main():
 torch.set_num_threads(2);start=time.monotonic();check=controls();rows=[]
 for index,family in enumerate(['independent','shared_input','shared_output','squares','cancellation']):
  torch.manual_seed(2870+index);rand=lambda *s:torch.randn(*s,dtype=torch.float64)
  u,v=rand(3,2,4),rand(3,2,4);c=rand(5,6)
  if family=='shared_input':u[:]=u[0]
  if family=='shared_output':c=c[:,:1]@rand(1,6)
  if family=='squares':v=u.clone()
  if family=='cancellation':u[2]=u[0];v[2]=v[0];c[:,5]=-c[:,0]
  u,v=normalize_bank(u,v);x,xv=rand(256,4),rand(1024,4);dx=.5*(rand(256,4)-x);dv=.5*(rand(1024,4)-xv)
  y=features(x,u,v)@c.T;yv=features(xv,u,v)@c.T;r=response_features(x,dx,u,v)@c.T;rv=response_features(xv,dv,u,v)@c.T
  for optimizer in ['adam','muon']:
   for seed in [0,1,'warm']:
    torch.manual_seed(2890+(0 if seed=='warm' else seed))
    initial=(u+.01*rand(*u.shape),v+.01*rand(*v.shape)) if seed=='warm' else (rand(*u.shape)/2,rand(*v.shape)/2)
    info,p=fit(x,y,dx,r,initial,steps=300,rate=.01 if seed=='warm' else .03,optimizer=optimizer)
    ve=float((evaluate(p,xv)-yv).norm()/yv.norm());re=float((response_features(xv,dv,p['U'],p['V'])@p['C'].T-rv).norm()/rv.norm())
    rows.append(dict(family=family,optimizer=optimizer,seed=seed,value_error=ve,response_error=re,objective=info['objective']))
  print(json.dumps(dict(family=family,results=rows[-6:])),flush=True)
 wins={op:sum(max(r['value_error'],r['response_error'])<.01 for r in rows if r['optimizer']==op and r['seed']!='warm') for op in ['adam','muon']}
 warm={op:all(max(r['value_error'],r['response_error'])<.01 for r in rows if r['optimizer']==op and r['seed']=='warm') for op in ['adam','muon']}
 selected=min(wins,key=lambda op:(not warm[op],-wins[op],sum(r['response_error'] for r in rows if r['optimizer']==op)))
 out=dict(controls=check,rows=rows,recovered_random_starts=wins,warm_all_pass=warm,selected_optimizer=selected,seconds=time.monotonic()-start,scope='Five known structures; two random starts plus near-teacher starts, matched initializations, one300step budget/rate per start class. Functional value/response test on independent Gaussian probes; not coefficient recovery, semantic identity or universal optimizer ranking.')
 Path(__file__).with_name('QUARTIC_RESPONSE_FIT_TOYS_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ['rows','controls']}),flush=True)
if __name__=='__main__':main()
