"""Two-stage algebraic proposal and continuous same-family refinement."""
from pathlib import Path
import json,time,torch,sys
from pairwise_product_toy_fixture import fixture
from pairwise_reader_graph import GROUPS,source_reads,price
from pencil_shared_discovery import propose_approximate
from pencil_shared_scalable import propose as propose_beam
from joint_orthogonal_private import JointOrthogonalPrivateMetric
from export_orthogonal_private import export
P=Path(__file__).parent;torch.set_num_threads(2);version=sys.argv[1] if len(sys.argv)>1 else 'V2';plan=json.loads((P/f'APPROXIMATE_PENCIL_PLAN_{version}.json').read_text());rows=[];start=time.monotonic()
for case in plan['cases']:
 clean,bases,private,_,_=fixture(case);d=clean.shape[-1];r=bases[0].shape[1];rng=torch.Generator().manual_seed(plan['seed_base']+case);noise=torch.randn(clean.shape,dtype=clean.dtype,generator=rng);noise=(noise+noise.transpose(-1,-2))/2
 for j in range(3):noise[2*j:2*j+2]*=clean[2*j:2*j+2].norm()/noise[2*j:2*j+2].norm()
 for level in plan['noise_levels']:
  target=clean+level*noise;row=dict(case=case,noise=level)
  try:
   proposal=(propose_beam(target,r,private[0].shape[1],plan['beam']) if plan.get('proposal_method')=='beam' else propose_approximate(target,r,private[0].shape[1]));initial=proposal['bases']+[torch.linalg.qr(v,mode='reduced').Q for v in proposal['private']]
   metric=JointOrthogonalPrivateMetric(target,[torch.cat([initial[a],initial[b]],1) for a,b in GROUPS]);params=[torch.nn.Parameter(v.detach().contiguous().clone()) for v in initial];initial_error=float(metric.loss(params,dense=True)[0].detach().sqrt());best=initial_error**2;saved=[v.detach().clone() for v in params];evaluations=0
   def closure():
    global best,saved,evaluations
    opt.zero_grad();loss=metric.loss(params,dense=True)[0];value=float(loss.detach());assert torch.isfinite(loss)
    if value<best:best=value;saved=[v.detach().clone() for v in params]
    evaluations+=1;scaled=loss/max(level,1e-6)**2;scaled.backward();
    for parameter in params:parameter.grad=parameter.grad.contiguous()
    return scaled
   if level>0:
    opt=torch.optim.LBFGS(params,lr=1.,max_iter=200,max_eval=250,history_size=50,line_search_fn='strong_wolfe',tolerance_grad=1e-10,tolerance_change=1e-14);opt.step(closure)
   with torch.no_grad():
    loss,cores=metric.loss(saved,dense=True);errors=[]
    for actual,true in zip(saved[:3],bases):
     a=torch.linalg.qr(actual,mode='reduced').Q;t=torch.linalg.qr(true,mode='reduced').Q;errors.append(float((a@a.T-t@t.T).norm()))
    template=dict(input_bases={str(j):saved[j] for j in range(3)},pairs={str(j):dict(a_linear=target.new_zeros(d),b_linear=target.new_zeros(d),a_bias=target.new_zeros(()),b_bias=target.new_zeros(())) for j in range(3)})
    I=torch.eye(d,dtype=target.dtype);graph,_=export(cores,metric.scales,template,I,I);z=torch.randn(32,d,dtype=target.dtype,generator=rng);actual=source_reads(z,graph)
    H=torch.cat([D@C@D.T+S@(A-W@C@W.T)@S.T for S,D,W,C,A in cores])*metric.scales.repeat_interleave(2)[:,None,None];dense=torch.einsum('ni,oij,nj->no',z,H,z);replay=float((actual-dense).norm()/dense.norm());assert replay<1e-8
    final=float(loss.sqrt());row.update(instrument=True,initial_error=initial_error,final_error=final,projector_errors=errors,source_multiplications=price(graph)['source_total_multiplications'],execution_replay=replay,evaluations=evaluations,proposal_stats=proposal['stats'],coefficient_pass=final<1e-8 if level==0 else final<=2*level,subspace_pass=max(errors)<=.1 if level==.01 else None)
  except (ValueError,RuntimeError,AssertionError) as error:row.update(instrument=False,failure=str(error))
  rows.append(row);print(json.dumps(row),flush=True)
summary={str(level):dict(valid=sum(r['instrument'] for r in rows if r['noise']==level),coefficient_passes=sum(r.get('coefficient_pass',False) for r in rows if r['noise']==level),subspace_passes=sum(bool(r.get('subspace_pass')) for r in rows if r['noise']==level)) for level in plan['noise_levels']}
result=dict(plan=plan,records=rows,summary=summary,seconds=time.monotonic()-start,predictions=dict(pred_a=summary['0.0']['coefficient_passes']==5,pred_b=all(summary[str(level)]['coefficient_passes']>=4 for level in plan['noise_levels'] if level>0),pred_c=summary['0.01']['subspace_passes']>=4),scope='Architecture-width support truncation, scored invariant block proposals, balanced shared edges, then local dense coefficient refinement. Exact graph export checked. Known directions used only to audit, never initialize. Tiny regular pencil cases; not native recovery.')
(P/f'APPROXIMATE_PENCIL_TOY_{version}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary),flush=True)
