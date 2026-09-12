"""Product Stiefel/sphere conjugate-gradient with projected vector transport."""
import time,torch
from quartic_preconditioner_v1 import precondition

def tangent(b,n,gb,gn):
 z=b.transpose(-1,-2)@gb
 return gb-b@((z+z.transpose(-1,-2))/2),gn-(gn*n).sum(-1,keepdim=True)*n

def retract(b,n,db,dn,step):
 q,r=torch.linalg.qr(b+step*db,mode='reduced');q=q*r.diagonal(dim1=-2,dim2=-1).sign()[:,None,:]
 newn=n+step*dn
 return q,newn/newn.norm(dim=-1,keepdim=True)

def inner(a,b):return sum((x*y).sum() for x,y in zip(a,b))

def fit(b,n,evaluate,divisor,max_steps=180,max_seconds=1000,callback=None):
 started=time.perf_counter();y,mix,gb,gn=evaluate(b,n,divisor,True);g=(gb,gn);z=precondition(b,n,*g);direction=tuple(-x for x in z);alpha=1.;history=[];reason='iteration_limit'
 for iteration in range(max_steps):
  norm2=float(inner(g,g));row=dict(iteration=iteration,objective=y,projected_gradient_norm=norm2**.5,seconds=time.perf_counter()-started);history.append(row)
  if callback:callback(row,b,n,mix)
  if norm2**.5<=1e-6:reason='stationary';break
  if time.perf_counter()-started>=max_seconds:reason='time_limit';break
  slope=float(inner(g,direction));pairing=float(inner(g,z));restarted=False
  if slope>=-.01*pairing:direction=tuple(-x for x in z);slope=-pairing;restarted=True
  accepted=False;trial_alpha=min(alpha*2,32.)
  for trial in range(20):
   bn,nn=retract(b,n,*direction,trial_alpha);yn,_=evaluate(bn,nn,divisor,False)
   if yn<=y+1e-4*trial_alpha*slope:accepted=True;break
   trial_alpha*=.5
  row.update(step_size=trial_alpha,line_trials=trial+1,restarted=restarted)
  if not accepted:reason='line_search_failed';break
  oldz=tangent(bn,nn,*z);oldd=tangent(bn,nn,*direction)
  b,n=bn.detach(),nn.detach();y,mix,gb,gn=evaluate(b,n,divisor,True);g=(gb,gn);z=precondition(b,n,*g)
  beta=max(0.,float(inner(g,tuple(a-c for a,c in zip(z,oldz))))/max(pairing,1e-30))
  direction=tuple(-a+beta*d for a,d in zip(z,oldd));alpha=trial_alpha
 history.append(dict(iteration=len(history),objective=y,projected_gradient_norm=float(inner(g,g).sqrt()),seconds=time.perf_counter()-started))
 return b,n,mix,history,reason
