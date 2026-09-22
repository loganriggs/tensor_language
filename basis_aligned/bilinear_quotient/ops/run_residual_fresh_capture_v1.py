#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_size pred_c_finite
"""Capture256x64 fresh states/reference16quartic outputs, no fitting.
Replay existing x/labels relative<1e-4; count16384; allfinite.
Prediction distribution tests are preregistered in RESIDUAL_FRESH_PLAN_V1.md.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);sys.path.insert(0,str(P))
 from native_quartic_branch import bilinear
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  x=torch.randn(256,64,1152);y=torch.randn(256,64,16);assert x.flatten(0,1).shape==(16384,1152) and y.flatten(0,1).shape==(16384,16)
  a=torch.randn(9,12,dtype=torch.float64);L=torch.randn(7,12,dtype=a.dtype);R=torch.randn_like(L);D=torch.randn(12,7,dtype=a.dtype);W=torch.randn(12,16,dtype=a.dtype)
  assert torch.allclose(bilinear(a,L,R,D)@W,((a@L.T)*(a@R.T))@(D.T@W))
  print(json.dumps(dict(actual_panel_shape=True,projected_teacher_replay=True)));return
 from native_feature_capture import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'RESIDUAL_FRESH_CAPTURE_V1.json';archive=P/'RESIDUAL_FRESH_STATES_V1.pt';assert not out.exists() and not archive.exists()
 meta=json.loads((P/'RESIDUAL_FRESH_ROWS_V1.json').read_text());tokenpath=P/'RESIDUAL_FRESH_TOKENS_V1.pt';assert hashlib.sha256(tokenpath.read_bytes()).hexdigest()==meta['tokens_sha256'];ids=torch.load(tokenpath,weights_only=True);assert ids.shape==(256,65)
 frozen=json.loads((P/'RESIDUAL_FRESH_CANDIDATES_V1.json').read_text())
 for file,sha in frozen.items():assert hashlib.sha256((P/file).read_bytes()).hexdigest()==sha
 model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17]
 writer=torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True)['writer'].cuda().float();uw=model.lm_head.weight@writer;readers=model.lm_head.weight.T@uw/uw.square().sum(0)
 def teacher(x):
  m=b17.lambdas[0]*bilinear(x,b16.mlp.Left.weight,b16.mlp.Right.weight,b16.mlp.Down.weight)
  return bilinear(m,b17.mlp.Left.weight,b17.mlp.Right.weight,b17.mlp.Down.weight)@readers
 oldids=torch.load(ROOT/'basis_aligned/bilinear_quotient/.rowcache/fineweb_n192_skip7000.pt',weights_only=True)[:4,:64]
 oldx=capture(model,oldids.cuda())['x16'].flatten(0,1);oldy=teacher(oldx)
 refx=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][1]['rows'][:256].cuda();refy=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'][1]['target'][:256].cuda()
 replay=dict(input=float((oldx.double()-refx.double()).norm()/refx.double().norm()),target=float((oldy.double()-refy.double()).norm()/refy.double().norm()))
 assert max(replay.values())<1e-4,replay
 xs=[];ys=[];times=[]
 for i in range(0,256,4):
  t=time.monotonic();x=capture(model,ids[i:i+4,:64].cuda())['x16'].flatten(0,1);y=teacher(x);xs.append(x.cpu());ys.append(y.cpu());times.append(time.monotonic()-t)
  if i%32==0:print(json.dumps(dict(prefixes=i+4,elapsed=time.monotonic()-start)),flush=True)
 x=torch.cat(xs);y=torch.cat(ys);pred=dict(pred_a_replay=max(replay.values())<1e-4,pred_b_size=x.shape==(16384,1152) and y.shape==(16384,16),pred_c_finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
 assert all(pred.values());torch.save(dict(rows=x,target=y,token_content_sha256=meta['token_content_sha256'],candidate_sha256=frozen,scope='Fresh256documentpanel, no fitting; scalar native quartic labels, no final-logit intervention.'),archive)
 result=dict(predictions=pred,replay=replay,seconds=time.monotonic()-start,batch_seconds=times,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),candidate_sha256=frozen,scope='Capture only; no accuracy result or circuit promotion.16384states,256documents,16coordinates. CPU evaluation follows.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
