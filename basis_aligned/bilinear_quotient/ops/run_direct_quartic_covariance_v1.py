#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_capture pred_b_identity pred_c_shift
"""MLP16 input statistics for quartic tensor metrics; no activation-target fitting."""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
PLAN=dict(panels=['fineweb_n96_skip1200.pt','fineweb_n192_skip7000.pt'],documents=32,tokens=64,batch=4,forwards=16)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_torch_save,guard_write
 out=P/'NATIVE_QUARTIC_COVARIANCE_V1.json';assert not out.exists();torch.set_num_threads(4);torch.set_grad_enabled(False);start=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model;block=model.transformer.h[16]
 panels=[];summaries=[];calls=0
 for panel in PLAN['panels']:
  ids=torch.load(BQ/'.rowcache'/panel,map_location='cpu',weights_only=True)[:32,:65];digest=hashlib.sha256(ids.numpy().tobytes()).hexdigest();values=[]
  def capture(_module,args):values.append(args[0].detach().double().reshape(-1,1152))
  handle=block.mlp.register_forward_pre_hook(capture)
  try:
   for startrow in range(0,32,4):
    tokens=ids[startrow:startrow+4].cuda();model(tokens[:,:-1],tokens[:,1:].contiguous());calls+=1
  finally:handle.remove()
  q=torch.cat(values);mean=q.mean(0);center=q-mean;cov=center.T@center/len(q);second=q.T@q/len(q)
  replay=float((q.float().double()-q).norm()/q.norm());identity=float((cov+mean[:,None]*mean[None,:]-second).norm()/second.norm());eig=torch.linalg.eigvalsh(cov)
  panels.append(dict(panel=panel,rows=q.float().cpu(),mean=mean.float().cpu(),covariance=cov.float().cpu(),second_moment=second.float().cpu(),token_sha256=digest))
  summaries.append(dict(panel=panel,rows=len(q),documents=32,token_sha256=digest,row_cast_error=replay,minimum_eigenvalue=float(eig.min()),maximum_eigenvalue=float(eig.max()),second_moment_identity=identity,covariance_trace=float(cov.trace()),mean_square=float(mean.square().sum()),eigenvalues=eig.cpu().tolist()))
 shift=float((panels[1]['covariance']-panels[0]['covariance']).norm()/panels[0]['covariance'].norm());instrument=calls==PLAN['forwards'] and all(x['rows']==2048 for x in summaries) and all(torch.isfinite(x['rows']).all().item() for x in panels)
 result=dict(plan=PLAN,calls=calls,panels=summaries,relative_covariance_shift=shift,predictions=dict(pred_a_capture=instrument,pred_b_identity=all(x['second_moment_identity']<1e-10 and x['minimum_eigenvalue']>=-1e-10*x['maximum_eigenvalue'] for x in summaries),pred_c_shift=shift>.01),seconds=time.perf_counter()-start,scope='Actual normalized MLP16 input in original residual coordinates. FP64 statistics saved FP32. Calibration/test panels fixed before collection. Gaussian covariance metric is an approximation to native higher moments; means/second moments saved separately. Not wholemodel polynomialization.')
 guard_torch_save(dict(panels=panels,scope=result['scope']),str(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt'));payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),label=out.name);out.write_text(payload);print(json.dumps({k:result[k] for k in ['calls','relative_covariance_shift','predictions','seconds']}))
if __name__=='__main__':main()
