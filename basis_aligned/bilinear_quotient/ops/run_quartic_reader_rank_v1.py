#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_rank pred_c_width
"""Two-panel native quartic derivative reader-rank bounds.
Integrity FP32/64<1e-3, spectral/trace<1e-8. Rank256floor>10%bothpanels;
rank512floor<=10%bothpanels. Necessarylocalbounds, notnaturalerror orcircuits.
No modelreplacement;32anchors, cachedpanels, exact1152-input Jacobians.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,panels=2,anchors_per_panel=16,ranks=[128,256,512,1024])));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from quartic_reader_span import jacobian
 from quartic_reader_rank_bound import analyze
 start=time.monotonic();out=P/'QUARTIC_READER_RANK_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 grams=[];precision=None
 for panel in panels:
  rows=panel['rows'];indices=torch.linspace(0,len(rows)-1,16).long();xs=rows[indices].cuda().float();g=torch.zeros(1152,1152,device='cuda',dtype=torch.float64)
  for x in xs:
   j=jacobian(teacher,x).double();g.add_(j.T@j)
   if precision is None:
    ref=jacobian([t.double() for t in teacher],x.double());precision=float((j-ref).norm()/ref.norm())
  grams.append(g)
 result,bases=analyze(grams,[128,256,512,1024]);trace_errors=[];projection_errors=[]
 for g,q,row in zip(grams,bases,result['panels']):
  trace_errors.append(abs(float(g.trace())-row['trace'])/row['trace'])
  direct=((g.trace()-(q[:,:256]*(g@q[:,:256])).sum())/g.trace()).clamp_min(0).sqrt();projection_errors.append(abs(float(direct)-row['floors']['256']))
 pred=dict(pred_a_integrity=precision<1e-3 and max(trace_errors+projection_errors)<1e-8,pred_b_rank=all(r['floors']['256']>.1 for r in result['panels']),pred_c_width=all(r['floors']['512']<=.1 for r in result['panels']))
 result.update(predictions=pred,native_fp32_fp64_jacobian_error=precision,trace_errors=trace_errors,projection_errors=projection_errors,seconds=time.monotonic()-start,scope='Necessary local Jacobian reader-rank bounds on two opened16-anchor panels, isotropic perturbations. Arbitrary whole-span rotation allowed; no natural-functional, semantic, or fullmodel adoption claim.')
 torch.save(dict(grams=[g.cpu() for g in grams],anchor_rows=torch.linspace(0,len(panels[0]['rows'])-1,16).long(),teacher_scale=scale),P/'QUARTIC_READER_GRAMS_V1.pt')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
