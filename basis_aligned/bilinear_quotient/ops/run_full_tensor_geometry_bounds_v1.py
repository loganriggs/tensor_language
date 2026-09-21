#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_residual pred_c_covariance
"""Full folded quadratic mode spectra. Zero model forwards/fits.
Predictions: trace consistency<1e-9, eig PSD>=-1e-10relative;
output rank128 and shared input rank512 each admit necessary error floor<.2.
Bounds necessary, not joint attainability or circuit evidence. Managed GPU only.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 sys.path.insert(0,str(P));from quadratic_mode_grams import grams,controls
 torch.set_num_threads(2);check=controls()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(controls=check,native_forwards=0)));return
 torch.set_grad_enabled(False);start=time.monotonic()
 snap=Path('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240');state=torch.load(snap/'pytorch_model.bin',map_location='cpu',weights_only=True)
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 D=w('transformer.h.17.mlp.Down.weight');L=w('transformer.h.17.mlp.Left.weight');R=w('transformer.h.17.mlp.Right.weight');Dprev=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];O=w('transformer.h.17.attn.c_proj.weight');S=torch.linalg.cholesky(torch.eye(1152,device='cuda',dtype=torch.float64)+Dprev@Dprev.T+O@O.T)
 data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);h=data['h'].cuda().double();hn=h/(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();centered=hn-hn.mean(0);M=centered.T@centered/len(hn);ev,V=torch.linalg.eigh(M);floor=1e-8*float(ev.mean());covroot=(V*ev.clamp_min(floor).sqrt())@V.T
 geometries={'folded_euclidean':S,'residual_euclidean':torch.eye(1152,device='cuda',dtype=torch.float64),'activation_covariance':covroot};records=[]
 def spectrum(e):
  e=e.clamp_min(0);tail=(1-e.cumsum(0)/e.sum()).clamp_min(0).sqrt();return dict(relative_error_floor={str(r):float(tail[r-1]) for r in (16,32,64,128,256,512,768,1024,1152)},necessary_rank_for_error={str(t):int((tail>t).sum())+1 for t in (.5,.2,.1,.05)})
 for geometry,Sg in geometries.items():
  C=RU@D;A=L@Sg;B=R@Sg;C/=C.norm();A/=A.norm();B/=B.norm();go,gi=grams(C,A,B);eo=torch.linalg.eigvalsh(go).flip(0);ei=torch.linalg.eigvalsh(gi).flip(0);trace=float(abs(go.trace()-gi.trace())/go.trace());negative=float(min(eo.min()/eo.max(),ei.min()/ei.max()));records.append(dict(geometry=geometry,trace_discrepancy=trace,minimum_relative_eigenvalue=negative,output=spectrum(eo),input=spectrum(ei)))
 pred=dict(pred_a_instrument=all(r['trace_discrepancy']<1e-9 and r['minimum_relative_eigenvalue']>=-1e-10 for r in records),pred_b_residual=records[1]['output']['relative_error_floor']['128']<.2,pred_c_covariance=records[2]['output']['relative_error_floor']['128']<.2 and records[2]['input']['relative_error_floor']['512']<.2)
 result=dict(controls=check,records=records,predictions=pred,covariance=dict(rows=len(h),mean_subtracted=True,eigenvalue_floor=floor,eigenvalues_floored=int((ev<floor).sum())),seconds=time.monotonic()-start,scope='Same full lastMLP quadratic and unembedding, three declared input metrics. Covariance is historical2048RMS-normalized h17 states, not fresh or document-independent. Covariance-shaped coefficient Frobenius is not exact text expected function error; biases, mean-generated lower-degree terms and normalization outside target.')
 (P/'FULL_TENSOR_GEOMETRY_BOUNDS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])))
if __name__=='__main__':main()
