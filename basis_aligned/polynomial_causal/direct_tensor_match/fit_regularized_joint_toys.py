"""Fixed small core-penalty sweep, preserving raw reconstruction and failures."""
from pathlib import Path
import json,math,time,torch
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS
from pairwise_exact_supports import supports
from regularized_joint_private import RegularizedJointMetric
P=Path(__file__).parent;torch.set_num_threads(2)
plan=json.loads((P/'REGULARIZED_JOINT_TOY_PLAN_V1.json').read_text());rows=[];states={};start=time.monotonic()
for case in range(5):
 T,bases,private,_,_=fixture(case);shared=[torch.cat([bases[a],bases[b]],1) for a,b in GROUPS];U=supports(T);shapes=[(u.shape[1],v.shape[1]) for u,v in zip(U,list(bases)+list(private))];expand=lambda ps:[u@v for u,v in zip(U,ps)]
 for rho in plan['rhos']:
  metric=RegularizedJointMetric(T,shared,rho)
  for optimizer in plan['optimizers']:
   for seed in plan['seeds']:
    rng=torch.Generator().manual_seed(42000+seed);params=[torch.nn.Parameter(torch.randn(shape,dtype=T.dtype,generator=rng)/shape[0]**.5) for shape in shapes]
    opt=torch.optim.Adam(params,lr=plan['rate']) if optimizer=='adam' else torch.optim.Muon(params,lr=plan['rate'],weight_decay=0.,adjust_lr_fn='match_rms_adamw');best=float('inf');key=f'{case}_{rho}_{optimizer}_{seed}';row=dict(key=key,case=case,rho=rho,optimizer=optimizer,seed=seed)
    try:
     for step in range(plan['steps']+1):
      opt.zero_grad();loss=metric.loss(expand(params))[0];value=float(loss.detach());assert math.isfinite(value)
      if value<best:best=value;saved=[v.detach().clone() for v in params];beststep=step
      if step==plan['steps']:break
      opt.param_groups[0]['lr']=plan['rate']*.5*(1+math.cos(math.pi*step/plan['steps']));loss.backward();opt.step()
     with torch.no_grad():
      explicit,cores=metric.loss(expand(saved),dense=True);raw=metric.reconstruction(expand(saved));replay=abs(float(explicit)-best);assert replay<1e-8
      norms=[float(c[3].norm()) for c in cores];den=[float(1-torch.linalg.eigvalsh(c[2].T@c[2]).square().max()) for c in cores]
     row.update(instrument=True,error=float(raw.sqrt()),penalized_loss=best,best_step=beststep,core_norms=norms,min_unregularized_denominators=den,loss_replay=replay);states[key]=dict(params=saved,supports=U)
    except (ValueError,RuntimeError,AssertionError) as error:row.update(instrument=False,error=None,failure=dict(step=step,error=str(error)))
    rows.append(row);(P/'REGULARIZED_JOINT_TOY_PARTIAL_V1.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n');print(json.dumps(row),flush=True)
summaries=[]
for rho in plan['rhos']:
 for optimizer in plan['optimizers']:
  group=[r for r in rows if r['rho']==rho and r['optimizer']==optimizer];valid=[r for r in group if r['instrument']];errors=[]
  for case in range(5):
   available=[r['error'] for r in valid if r['case']==case];errors.append(min(available) if available else None)
  score=math.exp(sum(math.log(max(r['error'],1e-8)) for r in group)/len(group)) if len(valid)==10 else None
  summaries.append(dict(rho=rho,optimizer=optimizer,best_restart_errors=errors,valid_runs=len(valid),recovered_cases=sum(e is not None and e<=plan['threshold'] for e in errors),geometric_mean_error=score))
eligible=[r for r in summaries if r['geometric_mean_error'] is not None];winner=min(eligible,key=lambda r:r['geometric_mean_error']) if eligible else None
out=dict(plan=plan,records=rows,summaries=summaries,winner=winner,recovery_pass=winner is not None and winner['recovered_cases']>=plan['required_cases'],seconds=time.monotonic()-start)
torch.save(states,P/'REGULARIZED_JOINT_TOY_STATES_V1.pt');(P/'REGULARIZED_JOINT_TOY_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summaries),flush=True)
