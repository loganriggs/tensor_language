import json,time
from pathlib import Path
import torch
from arithmetic_dag import DAG
from trainable_arithmetic_dag import TrainableDAG
P=Path(__file__).resolve().parent

def planted(case):
 d=DAG(degree_limit=4);x=[d.input(i) for i in range(4)];one=d.constant();assert one==d.constant()
 gen=torch.Generator().manual_seed(99)
 def linear(nodes):return d.linear(zip(nodes,(torch.randn(len(nodes),generator=gen)*.4+.1).tolist()))
 a,b,c,e=[linear(x+[one]) for _ in range(4)]
 if case=='affine_quadratic_residual':features=[d.product(a,b),c,one]
 elif case=='shared_linear_sum':
  t=d.linear([(b,1),(c,1)]);features=[d.product(a,t),d.product(e,t),one]
 elif case=='square_quadratic':
  q=d.linear([(d.product(a,b),1),(d.product(c,e),1)]);features=[d.product(q,q),one]
 elif case=='shared_quadratic_branches':
  q=d.product(a,b);features=[d.product(q,d.product(c,c)),d.product(q,d.product(e,e)),one]
 else:
  q=d.product(a,b);r=d.product(q,c);features=[q,r,d.product(r,e),a,one]
 out=[linear(features) for _ in range(2)];return d,out

def main():
 start=time.perf_counter();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
 families=['affine_quadratic_residual','shared_linear_sum','square_quadratic','shared_quadratic_branches','mixed_degree_skips'];records=[];replays=[];exports=[];graderrors=[];degree_checks=[]
 for family in families:
  d,out=planted(family);teacher=TrainableDAG(d,out);gen=torch.Generator().manual_seed(314);x=torch.randn(1024,4,generator=gen);validation=torch.randn(4096,4,generator=gen)
  with torch.no_grad():y=teacher(x);yv=teacher(validation)
  replays.append(float((y-d.evaluate(out,x)).norm()/y.norm()))
  # Shared-node autodiff checked against a simultaneous parameter directional derivative.
  params=list(teacher.parameters());directions=[torch.randn(p.shape,generator=gen) for p in params];loss=teacher(x[:17]).square().mean();grads=torch.autograd.grad(loss,params);analytic=sum((g*v).sum() for g,v in zip(grads,directions));original=[p.detach().clone() for p in params];values=[]
  for sign in [1,-1]:
   with torch.no_grad():
    for p,v,o in zip(params,directions,original):p.copy_(o+sign*1e-6*v)
    values.append(teacher(x[:17]).square().mean())
  numeric=(values[0]-values[1])/2e-6;graderrors.append(float(abs(numeric-analytic)/max(1.,abs(float(analytic)))))
  with torch.no_grad():
   for p,o in zip(params,original):p.copy_(o)
  n=len(d.nodes);highest=max(d.reachable(out),key=lambda k:d.degrees[k]);rejected=False
  try:d.product(highest,d.product(highest,highest))
  except ValueError:rejected=True
  # Nested expression may add an admissible intermediate before the rejected outer node.
  degree_checks.append(rejected and max(d.degrees)<=4)
  for seed in [0,1]:
   torch.manual_seed(seed);student=TrainableDAG(d,out)
   with torch.no_grad():
    for p in student.parameters():p.copy_(torch.randn_like(p)*.3)
   opt=torch.optim.Adam(student.parameters(),lr=.03);scale=y.square().mean();best=float('inf');checkpoint=None
   for step in range(1500):
    opt.zero_grad();loss=(student(x)-y).square().mean()/scale
    if float(loss.detach())<best:best=float(loss.detach());checkpoint={k:v.detach().clone() for k,v in student.state_dict().items()}
    loss.backward();opt.step()
   with torch.no_grad():final=float((student(x)-y).square().mean()/scale)
   if final<best:best=final;checkpoint={k:v.detach().clone() for k,v in student.state_dict().items()}
   student.load_state_dict(checkpoint)
   with torch.no_grad():pred=student(validation);err=float((pred-yv).norm()/yv.norm());export,eout=student.export();replay=float((export.evaluate(eout,validation)-pred).norm()/max(float(pred.norm()),1e-30));exports.append(replay)
   row=dict(family=family,seed=seed,training_best_relative_error=best**.5,training_final_relative_error=final**.5,validation_relative_error=err,export_replay=replay,price=student.price(),degree=max(d.degrees[k] for k in out));records.append(row);print(row,flush=True)
 # Atomic rejection tested directly rather than composing two API calls.
 check=DAG(degree_limit=4);a=check.input(0);q=check.product(a,a);f=check.product(q,q);before=len(check.nodes)
 try:check.product(f,a);raise AssertionError('accepted degree five')
 except ValueError:assert len(check.nodes)==before
 good=sum(min(r['validation_relative_error'] for r in records if r['family']==f)<.01 for f in families)
 result=dict(records=records,oracle_replay=max(replays),gradient_relative_discrepancy=max(graderrors),export_replay=max(exports),families_below_one_percent=good,predictions=dict(pred_a=max(replays+exports)<1e-10 and max(graderrors)<1e-6,pred_b=all(degree_checks),pred_c=good>=4),seconds=time.perf_counter()-start,scope='Known topology capacity controls with random continuous fitting. Not topology discovery, optimizer comparison, text OOD, or identified semantics.')
 (P/'TRAINABLE_DAG_CHECK_V1.json').write_text(json.dumps(result,indent=2)+'\n');print('SUMMARY',result['predictions'],result['seconds'],flush=True)
if __name__=='__main__':main()
