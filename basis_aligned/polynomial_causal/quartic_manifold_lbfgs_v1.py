"""Experimental projected-transport limited-memory BFGS with safeguarded descent."""
import time,torch
from quartic_manifold_cg_v1 import tangent,retract,inner

def combine(a,b,scale):return tuple(x+scale*y for x,y in zip(a,b))
def positive(s,y):
 curvature=float(inner(s,y));norm=float((inner(s,s)*inner(y,y)).sqrt())
 return curvature>max(1e-30,1e-8*norm)
def fit(b,n,evaluate,divisor,max_steps=180,max_seconds=1000,callback=None):
 started=time.perf_counter();value,mix,gb,gn=evaluate(b,n,divisor,True);g=(gb,gn);pairs=[];history=[];reason='iteration_limit'
 for iteration in range(max_steps):
  norm2=float(inner(g,g));row=dict(iteration=iteration,objective=value,projected_gradient_norm=norm2**.5,memory_pairs=len(pairs),seconds=time.perf_counter()-started);history.append(row)
  if callback:callback(row,b,n,mix)
  if norm2**.5<=1e-6:reason='stationary';break
  if time.perf_counter()-started>=max_seconds:reason='time_limit';break
  q=g;coeff=[]
  for s,y in reversed(pairs):
   alpha=float(inner(s,q)/inner(s,y));coeff.append(alpha);q=combine(q,y,-alpha)
  gamma=float(inner(*pairs[-1])/inner(pairs[-1][1],pairs[-1][1])) if pairs else 1.
  r=tuple(x*max(1e-6,min(gamma,1e6)) for x in q)
  for (s,y),alpha in zip(pairs,reversed(coeff)):
   beta=float(inner(y,r)/inner(s,y));r=combine(r,s,alpha-beta)
  direction=tangent(b,n,*tuple(-x for x in r));slope=float(inner(g,direction));reset=False
  if slope>=-.001*norm2:direction=tuple(-x for x in g);slope=-norm2;pairs=[];reset=True
  accepted=False;alpha=1.
  for trial in range(20):
   bn,nn=retract(b,n,*direction,alpha);newvalue,_=evaluate(bn,nn,divisor,False)
   if newvalue<=value+1e-4*alpha*slope:accepted=True;break
   alpha*=.5
  row.update(step_size=alpha,line_trials=trial+1,history_reset=reset)
  if not accepted:reason='line_search_failed';break
  transported=[]
  for s,y in pairs:
   sn=tangent(bn,nn,*s);yn=tangent(bn,nn,*y)
   if positive(sn,yn):transported.append((sn,yn))
  oldg=tangent(bn,nn,*g);s=tangent(bn,nn,*tuple(alpha*x for x in direction))
  b,n=bn.detach(),nn.detach();value,mix,gb,gn=evaluate(b,n,divisor,True);g=(gb,gn);y=combine(g,oldg,-1.)
  if positive(s,y):transported.append((s,y))
  pairs=transported[-8:]
 history.append(dict(iteration=len(history),objective=value,projected_gradient_norm=float(inner(g,g).sqrt()),memory_pairs=len(pairs),seconds=time.perf_counter()-started))
 return b,n,mix,history,reason
