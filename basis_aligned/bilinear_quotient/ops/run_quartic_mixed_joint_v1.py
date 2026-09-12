#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;240mixed-graph optimization steps,128cached validationvectors.
"""pred_a initial replay<=1e-8 and monotonicobjective;
pred_b coefficient capture gain>=10% over mixed start;
pred_c projected gradient<=1e-6. 240steps/1200fitsec, no textfit.
32rank16quadratics,32productedges,592704floats. A standalone graph needs64
edge indices; shared scorer serialization stores1056 lookup and32 selection indices.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from sparse_path_stability_atlas_v1 import digest
from quartic_selected_edges_v1 import features
from mixed_quartic_objective_v1 import make_objective
from quartic_manifold_lbfgs_v1 import fit
from composed_quartic_contraction_v1 import contract
STEM='QUARTIC_MIXED_JOINT_V1'
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 price=json.loads((P/'QUARTIC_SELECTED_EDGES_PRICE_V1_RESULT.json').read_text())
 assert price['pred_a'] and price['pred_b'] and price['pred_c']
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,optimization_steps=240,maximum_line_search=20,fitted_floats=592704)));return
 out=P/(STEM+'_RESULT.json');ap=P/(STEM+'_PROGRAMS.pt');assert not out.exists() and not ap.exists();signal.alarm(1500)
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 native=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')]
 scale=float(state['transformer.h.17.lambdas'][0])
 p=torch.load(P/'QUADRATIC_PRODUCT_CORE_LEARNED_V1_PROGRAMS.pt',weights_only=True)['programs'][0];assert p['seed']==11511
 b=p['input_readers'].cuda();n=p['inner_weights'].cuda();edges=p['pairs'][:,p['selected']].cuda();writer=p['output_writers'].cuda()
 native[-1]=(metric@writer).T@native[-1]
 divisor=torch.load(P/'COUPLED_QUARTIC_LBFGS_V1_PROGRAM.pt',weights_only=True)['divisor']
 evaluate=make_objective(edges,lambda slots:contract(slots,*native,scale))
 initial,mixing=evaluate(b,n,divisor);replay=abs(initial-price['initial_objective']);assert replay<=1e-8
 x=torch.load(P/'QUARTIC_GROUP_PORTS_V1_PORTS.pt',weights_only=True)['input16'].double().cuda()
 pre=torch.load(P/'NATIVE_RELATION_OUTPUT_FRESH_V1_ENDPOINTS.pt',weights_only=True)['ports']['pre'].double().cuda();den=pre.square().mean(-1)+torch.finfo(torch.float32).eps
 reference=torch.load(P/'QUARTIC_GROUP_BOUNDARY_V1_WRITES.pt',weights_only=True)['lifted'][0].cuda()
 def error(b,n,mix):return float((features(b,n,edges,x)@mix@writer.T/den[:,None]-reference).norm()/reference.norm())
 before=error(b,n,mixing)
 source=json.loads((P/'QUADRATIC_PRODUCT_CORE_LEARNED_V1_RESULT.json').read_text())['reports'][0]
 write_replay=abs(before-source['sparse_native_error']);assert write_replay<=1e-8
 def callback(row,b,n,mixing):
  if row['iteration']%10==0:
   (P/(STEM+'_PROGRESS.json')).write_text(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
 b,n,mixing,history,reason=fit(b,n,evaluate,divisor,max_steps=240,max_seconds=1200,callback=callback)
 after=error(b,n,mixing)
 torch.save(dict(programs=[dict(seed=11511,input_readers=b.cpu(),inner_weights=n.cpu(),output_writers=writer.cpu(),pairs=p['pairs'],selected=p['selected'],sparse_mixing=mixing.cpu(),divisor=divisor)]),ap)
 monotone=all(y['objective']<=z['objective']+1e-10 for z,y in zip(history,history[1:]))
 gain=history[-1]['objective']/initial-1
 result={'pred_a':max(replay,write_replay)<=1e-8 and monotone,'pred_b':gain>=.1,'pred_c':history[-1]['projected_gradient_norm']<=1e-6}
 result.update(dict(history=history,termination=reason,coefficient_gain_fraction=gain,initial_objective=initial,
   reports=[dict(seed=11511,baseline_native_error=source['baseline_native_error'],fit_initial_native_error=before,sparse_native_error=after)],
   coefficient_replay_error=replay,write_replay_error=write_replay,artifact_sha256=digest(ap),wall_seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
   scope='Fixed learned32edge graph jointreader fit, twooutputgroup. Shared effect scorer compares original LBFGS V1 squares; fit_initial_native_error is separately the mixed starting program. No textfit, OOD or selective-circuit claim.'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='history'}),flush=True);assert result['pred_a']
if __name__=='__main__':main()
