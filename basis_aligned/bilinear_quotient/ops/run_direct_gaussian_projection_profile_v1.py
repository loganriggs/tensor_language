#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_precision pred_b_gradients pred_c_cost
"""Native Gaussian quadratic projection profile, MLP16 -> MLP17 -> unembedding.
pred_a_precision: FP32/64 cross relative error <1e-3 on both metrics.
pred_b_gradients: finite nonzero reader gradients and scaling tripwire <1e-4.
pred_c_cost: four-product FP32 forward/backward <10s, peak allocation <16GiB.
Null: contractions too expensive or unstable. No fitting. Prospective student
price 34560 scalars,4 products including fixed rank8 linear branch and constant.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(metrics=['centered','spherical'],products=4,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P))
 from gaussian_projected_quadratic_cross import ProjectedQuadratic
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'NATIVE_GAUSSIAN_PROJECTION_PROFILE_V1.json';assert not out.exists()
 source=torch.load(P/'REGULARIZED_CENTERED_EXPORT_V1.pt',weights_only=True);s=source['programs'][.001];panel=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True)
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/source['teacher_scale'],w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 del state,ru
 records=[]
 for metric in PLAN['metrics']:
  cov=panel['covariance'].cuda().double()
  if metric=='spherical':cov=torch.eye(len(cov),device='cuda',dtype=torch.float64)*cov.trace()/len(cov)
  values={};row=dict(metric=metric)
  for dtype in [torch.float32,torch.float64]:
   label=str(dtype).split('.')[-1];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
   op=ProjectedQuadratic([t.to(dtype) for t in teacher],panel['mean'].cuda().to(dtype),cov.to(dtype));torch.cuda.synchronize();row[label+'_setup_seconds']=time.perf_counter()-start
   a=s['quadratic_left'].cuda().to(dtype).requires_grad_();b=s['quadratic_right'].cuda().to(dtype).requires_grad_();torch.cuda.synchronize();start=time.perf_counter()
   cross=op.cross(a,b);loss=cross.square().sum();ga,gb=torch.autograd.grad(loss,[a,b]);torch.cuda.synchronize();row[label+'_step_seconds']=time.perf_counter()-start;row[label+'_peak_gib']=torch.cuda.max_memory_allocated()/2**30;row[label+'_finite_gradients']=bool(torch.isfinite(ga).all() and torch.isfinite(gb).all() and ga.norm()>0 and gb.norm()>0);values[label]=cross.detach().double().cpu()
   if dtype==torch.float32:
    with torch.no_grad():row['scaling_tripwire_error']=float((op.cross(2*a[:1],b[:1])-2*cross[:,:1]).norm()/cross[:,:1].norm())
   del op,a,b,cross,loss,ga,gb;torch.cuda.empty_cache()
  row['precision_relative_error']=float((values['float32']-values['float64']).norm()/values['float64'].norm());records.append(row);print(json.dumps(row),flush=True)
 pred=dict(pred_a_precision=all(r['precision_relative_error']<1e-3 for r in records),pred_b_gradients=all(r['float32_finite_gradients'] and r['float64_finite_gradients'] and r['scaling_tripwire_error']<1e-4 for r in records),pred_c_cost=all(r['float32_step_seconds']<10 and r['float32_peak_gib']<16 for r in records))
 out.write_text(json.dumps(dict(plan=PLAN,records=records,predictions=pred,scope='Native contraction profiling only. FP64 references the same FP32-formed teacher factors; tests contraction precision, not checkpoint folding precision.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
