"""Exact DAG rewrite controls and CPU benchmark; source exports unchanged."""
import json,time,statistics
import torch
from shifted_pair_cp import compile_program,evaluate
from lean_conditional_cp import evaluate as baseline,price
from audit_conditional_residual_accounting import P,load,SCALE

def main():
 torch.set_num_threads(2);torch.manual_seed(2209220206);torch.set_grad_enabled(False);controls=[]
 for family in ['random','zero_correction','repeated_factors','large_correction','constant_only']:
  p=dict(input_projection=torch.randn(3,5,dtype=torch.float64),factors=[torch.randn(7,3,dtype=torch.float64) for _ in range(4)],biases=[torch.randn(7,dtype=torch.float64) for _ in range(4)],coefficients=torch.randn(2,7,dtype=torch.float64),constant=torch.randn(2,dtype=torch.float64),pair_weights=torch.randn(2,7,dtype=torch.float64),matching=0)
  if family=='zero_correction':p['pair_weights'].zero_()
  if family=='repeated_factors':p['factors']=[p['factors'][0]]*4
  if family=='large_correction':p['pair_weights']*=1000
  if family=='constant_only':p['factors']=[torch.zeros_like(f) for f in p['factors']]
  x=torch.randn(23,5,dtype=torch.float64)
  for matching in range(3):
   p['matching']=matching;y=baseline(p,x);z=evaluate(compile_program(p),x);err=float((y-z).norm()/y.norm());assert err<1e-10
   controls.append(dict(family=family,matching=matching,error=err))
 x=torch.load(P/'RESIDUAL_FRESH_STATES_V1.pt',weights_only=True)['rows'];rows=[]
 for seed in [1001,1002]:
  p,h=load(f'LEAN_CONDITIONAL_CP_SEED{seed}_RANK256_V1.pt');p['coefficients']/=SCALE;p['constant']/=SCALE
  q=compile_program(p);yd=torch.cat([baseline(p,xx.double()) for xx in x.split(2048)]);zd=torch.cat([evaluate(q,xx.double()) for xx in x.split(2048)]);double_err=float((yd-zd).norm()/yd.norm());assert double_err<1e-12
  def cast(a):return {k:[v.float() for v in vs] if isinstance(vs,list) else vs.float() if isinstance(vs,torch.Tensor) else vs for k,vs in a.items()}
  pf=cast(p);qf=cast(q);yf=torch.cat([baseline(pf,xx) for xx in x.split(2048)]);zf=torch.cat([evaluate(qf,xx) for xx in x.split(2048)]);ferr=float((yf-zf).norm()/yf.norm());assert ferr<1e-5
  timings=[]
  for batch,reps in [(1,100),(64,10),(2048,1)]:
   xx=x[:batch];ts=[[],[]];functions=[lambda:baseline(pf,xx),lambda:evaluate(qf,xx)]
   for f in functions:f()
   for r in range(7):
    for i in ([0,1] if r%2 else [1,0]):
     start=time.perf_counter()
     for _ in range(reps):functions[i]()
     ts[i].append((time.perf_counter()-start)/reps)
   timings.append(dict(batch=batch,baseline=ts[0],rewritten=ts[1],speedup=statistics.median(ts[0])/statistics.median(ts[1])))
  oldprice=price(p);newprice=dict(oldprice);newprice['coefficient_multiplications']-=2*p['coefficients'].shape[1]
  rows.append(dict(seed=seed,source_sha256=h,float64_replay=double_err,float32_replay=ferr,cost_before=oldprice,cost_after=newprice,timings=timings));print(seed,double_err,ferr,[r['speedup'] for r in timings],flush=True)
 (P/'SHIFTED_PAIR_CP_V1.json').write_text(json.dumps(dict(controls=controls,rows=rows,scope='Exact real-arithmetic rewrite, compiled constants in float64 then cast. Warm eagerCPU2threads7rounds. No source mutation, fitting, newaccuracyclaim, candidateexportorqueuedhelperchange.'),indent=2)+'\n')
if __name__=='__main__':main()
