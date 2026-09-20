#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact pred_b_rank64 pred_c_rank256
"""MIDPOINT_SPECTRUM_PLAN_V1: pred_a_exact toy<1e-12, PSD>-1e-10, trace<1e-8;
pred_b_rank64 coefficient errorfloor<.5; pred_c_rank256 errorfloor<.2.
Null broad spectrum; price zero forwards, native QR+4608² FP64 contractions.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(native_forwards=0,largest_matrix=[4608,4608],output=1152)));return
 import torch
 sys.path.insert(0,str(P))
 from midpoint_spectrum import output_gram,spectrum,toy_check
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'MIDPOINT_SPECTRUM_V1.json';assert not out.exists();start=time.perf_counter();toy=toy_check()
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17]
 _,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced')
 factors=(ru@b17.mlp.Down.weight.double(),b17.mlp.Left.weight.double(),b17.mlp.Right.weight.double(),b17.lambdas[0].double()*b16.mlp.Down.weight.double())
 assert [list(t.shape) for t in factors]==[[1152,4608],[4608,1152],[4608,1152],[1152,4608]]
 scales=[float(t.norm()) for t in factors];C,L,R,D=[t/s for t,s in zip(factors,scales)]
 gram=output_gram((C,L,R,D));e,negative=spectrum(gram)
 # Independent explicit input-factor contraction; no expanded output tensor.
 LD=L@D;RD=R@D;CC=C.T@C
 norm=(CC*((L@L.T)*(RD@RD.T)+(R@R.T)*(LD@LD.T)+2*(L@R.T)*(RD@LD.T))).sum()
 trace_error=float((gram.trace()-norm).abs()/norm.abs())
 floors={str(r):float((e[r:].sum()/e.sum()).sqrt()) for r in [4,16,32,64,128,256,512,1024,1152]}
 pred=dict(pred_a_exact=max(toy.values())<1e-12 and negative>-1e-10 and trace_error<1e-8,pred_b_rank64=floors['64']<.5,pred_c_rank256=floors['256']<.2)
 result=dict(predictions=pred,toy=toy,trace_relative_error=trace_error,min_eigenvalue_over_energy=negative,coefficient_error_lower_bounds=floors,eigenvalues=e.cpu().tolist(),factor_scales=scales,normalized_tensor_squared_norm=float(norm),seconds=time.perf_counter()-start,scope='Independent midpoint/channel-product coefficient metric; not native functional floor, circuit identification, or fitted compression.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='eigenvalues'},indent=2))
if __name__=='__main__':main()
