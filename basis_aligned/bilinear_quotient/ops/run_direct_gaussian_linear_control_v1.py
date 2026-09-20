#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_derivative pred_b_prediction pred_c_export
"""Fixed-reader Gaussian linear correction; GAUSSIAN_LINEAR_CONTROL_PLAN_V1.
pred_a_derivative: native derivative vs central difference <1e-3, finite.
pred_b_prediction: centered winner error improves >=.005 on panel2.
pred_c_export: 34560 scalars,4products, FP32 export replay <1e-5.
Null: fixed linear branch is not the main limitation. Both covariance metrics.
Profile first JVP before continuation; stop if >10s or peak allocation >16GiB.
No empirical output fitting; exact weights and input moments only.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(metrics=['centered','spherical'],reader_rank=8,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 from disk_guard import guard_torch_save
 sys.path.insert(0,str(P))
 from gaussian_linear_projection import directions
 from gaussian_quartic_mean import native_mean
 from audit_centered_compact import evaluate
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.json';assert not out.exists();start=time.perf_counter()
 source=torch.load(P/'NATIVE_GAUSSIAN_QUADRATIC_FIT_V1.pt',weights_only=True);result=json.loads((P/'NATIVE_GAUSSIAN_QUADRATIC_FIT_V1.json').read_text());panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];targets=[t.cuda().double() for t in torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True)['targets']];mu=panels[0]['mean'].cuda().double();cov=panels[0]['covariance'].cuda().double()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/source['teacher_scale'],w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];teacher=[t.double() for t in teacher]
 del state,ru
 rows=[];exports={}
 for winner in result['winners']:
  metric=winner['metric'];key=tuple(winner[k] for k in ['metric','warmstart','optimizer']);s={k:t.cuda().double() for k,t in source['programs'][key].items()};M=cov if metric=='centered' else torch.eye(len(mu),device='cuda',dtype=torch.float64)*cov.trace()/len(mu);A=s['linear_reader'];vectors=A@M
  torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();begin=time.perf_counter();first=directions(teacher,mu,M,vectors[:1]);torch.cuda.synchronize();elapsed=time.perf_counter()-begin;peak=torch.cuda.max_memory_allocated()/2**30
  assert elapsed<10 and peak<16,'JVP profile exceeded registered resource budget'
  eps=1e-4*mu.norm()/vectors[0].norm();fd=(native_mean(teacher,mu+eps*vectors[0],M)-native_mean(teacher,mu-eps*vectors[0],M))/(2*eps);check=float((first[:,0]-fd).norm()/first.norm());X=torch.cat([first,directions(teacher,mu,M,vectors[1:])],1);G=A@M@A.T;W=torch.linalg.solve(G,X.T).T;old=s['linear_writer'];gain=float((((old@G)*old).sum()-2*(old*X).sum())-(((W@G)*W).sum()-2*(W*X).sum()));model={**s,'linear_writer':W};archive={k:t.float().cpu() for k,t in model.items()};loaded={k:t.cuda() for k,t in archive.items()};scores=[];replays=[]
  for panel,y in zip(panels,targets):
   x=panel['rows'].cuda().double();before=evaluate(s,x);after=evaluate(loaded,x);scores.append(dict(before_error=float((before-y).norm()/y.norm()),after_error=float((after-y).norm()/y.norm()),variation=float(((after-after.mean(0))-(y-y.mean(0))).norm()/(y-y.mean(0)).norm())));replays.append(float((after[:32]-evaluate(model,x[:32])).norm()/after[:32].norm()))
  row=dict(metric=metric,first_jvp_seconds=elapsed,peak_gib=peak,finite_difference_error=check,finite=bool(torch.isfinite(X).all()),gram_condition=float(torch.linalg.cond(G)),gaussian_linear_gain=gain,normal_equation_error=float((W@G-X).norm()/X.norm()),panels=scores,archive_replay=max(replays),scalar_count=sum(t.numel() for t in archive.values()),products=4);rows.append(row);exports[metric]=archive;print(json.dumps(row),flush=True)
 pred=dict(pred_a_derivative=all(r['finite_difference_error']<1e-3 and r['finite'] for r in rows),pred_b_prediction=rows[0]['panels'][1]['after_error']<=rows[0]['panels'][1]['before_error']-.005,pred_c_export=all(r['scalar_count']==34560 and r['archive_replay']<1e-5 for r in rows));guard_torch_save(dict(programs=exports,teacher_scale=source['teacher_scale']),str(P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=rows,predictions=pred,seconds=time.perf_counter()-start,scope='Fixed linear reader, quadratic branch and constant. Gaussian-optimal writer in fixed reader span, not globally optimal rank8 linear approximation. Same reused diagnostic panels.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
