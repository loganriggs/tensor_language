#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_missing pred_c_concentrated
"""Native quartic response in the learned32-bank reader nullspace.
Integrity FP32/64Jacobian<1e-3,studentnullinvariance<1e-5relative.
MissingJacobian aggregate>10%and12/16anchors>10%; <=32directions capture90%.
Null: missingresponse small ordiffuse. Fixed656product903168coefficient
program. No nativebehavior, semantic or OODadoption claim.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,anchors=16,reader_limit=256,input_width=1152)));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from quartic_reader_span import forward,jacobian,reader_basis
 from empirical_quartic_dictionary import evaluate
 start=time.monotonic();out=P/'QUARTIC_READER_SPAN_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];rows=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'];indices=torch.linspace(0,len(rows)-1,16).long();xs=rows[indices].cuda().float()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')]
 archive=torch.load(P/'WIDE_NATIVE_QUARTIC_32_INHERITED_LONG_V1.pt',weights_only=True);program={k:v.cuda().double() for k,v in archive.items()};q=reader_basis(program['U'],program['V']);gram=torch.zeros(1152,1152,device='cuda',dtype=torch.float64);den=0.;records=[];torch.manual_seed(4801)
 for index,x in zip(indices.tolist(),xs):
  j=jacobian(teacher,x).double();lost=j-(j@q)@q.T;gram.add_(lost.T@lost);den+=float(j.square().sum());fraction=float(lost.norm()/j.norm())
  if not records:
   reference=jacobian([t.double() for t in teacher],x.double());precision=float((j-reference).norm()/reference.norm())
  z=torch.randn_like(x,dtype=torch.float64);z=z-q@(q.T@z);z=z/z.norm()*(.01*x.norm());end=x.double()+z
  y0=evaluate(program,x[None].double());y1=evaluate(program,end[None]);invariance=float((y1-y0).norm()/y0.norm());change=forward(teacher,end[None].float())-forward(teacher,x[None]);teacher_change=float(change.norm()/forward(teacher,x[None]).norm())
  records.append(dict(row=index,unread_jacobian_fraction=fraction,student_finite_invariance=invariance,teacher_finite_relative_change=teacher_change))
 eigen=torch.linalg.eigvalsh((gram+gram.T)/2).flip(0).clamp_min(0);cumulative=eigen.cumsum(0)/eigen.sum();rank90=int(torch.searchsorted(cumulative,torch.tensor(.9,device='cuda',dtype=torch.float64)))+1;aggregate=float((gram.trace()/den).sqrt());pred=dict(pred_a_integrity=precision<1e-3 and max(r['student_finite_invariance'] for r in records)<1e-5,pred_b_missing=aggregate>.1 and sum(r['unread_jacobian_fraction']>.1 for r in records)>=12,pred_c_concentrated=rank90<=32)
 result=dict(predictions=pred,records=records,reader_rank=q.shape[1],aggregate_unread_jacobian_fraction=aggregate,unread_rank90=rank90,top32_unread_energy_fraction=float(cumulative[31]),native_fp32_fp64_jacobian_error=precision,seconds=time.monotonic()-start,scope='Pure quartic target; sixteen calibration anchors; isotropic local derivatives and artificial finite nullspace perturbations. Fixed-reader restriction only, not a global rank lower bound or natural-input error bound. No new semantic or OOD adoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
