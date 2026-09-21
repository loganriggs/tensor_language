#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_transfer_gap pred_c_ordering
"""Opened48docs, no fit: exact training-metric error on token-matched source swaps.
Native replay<1e-5; learned centeredquadratic responseerror>2*.0326400716
in>=3of4cohorts; learned<=fixed on>=3of4cohorts. Null: fitting geometry already
transfers before final normalization. Same frozen3686product candidates.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(captures=48,fit=False,programs=2)));return
 import torch
 import torch.nn.functional as F
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'FINITE_RESPONSE_GEOMETRY_AUDIT_V1.json';assert not out.exists()
 tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);donors=json.loads((P/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_DONORS_V1.json').read_text());assert hashlib.sha256((P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt').read_bytes()).hexdigest()==donors['token_sha256']
 model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17]
 RU=torch.load(P/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True)['R_U'].cuda().double();cal=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True)['h'].cuda().double()
 norm=lambda x:x/(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt()
 mu=norm(cal).mean(0);L,R,D=[getattr(b17.mlp,k).weight.double() for k in ['Left','Right','Down']]
 programs={name:{k:v.cuda().double() for k,v in torch.load(P/f'FULL_QUADRATIC_FINITE_{name.upper()}_PROGRAM_V1.pt',weights_only=True).items()} for name in ['fixed','learned']}
 teacher=lambda x:((x@L.T)*(x@R.T))@D.T
 def student(x,p):return ((x@p['a'].T)*(x@p['b'].T))@p['writer'].T+x@p['linear'].T+p['bias']
 records=[];checks=[]
 for domain,docs in tokens.items():
  hh=[];sources=[]
  for row in docs:
   cache=capture(model,row[None,:256].cuda());h=cache['h17'];checks.append(float((h+b17.mlp(F.rms_norm(h,(1152,)))-cache['final']).norm()/cache['final'].norm()));hh.append(h.flatten(0,1)[16:255]);sources.append((b17.lambdas[0]*(cache['m16']-b16.mlp.Down_bias)).flatten(0,1)[16:255])
  hh=torch.cat(hh).double();sources=torch.cat(sources).double();mapping=torch.tensor(donors['domains'][domain]['maps']['same_cohort'],device='cuda');cohorts=torch.tensor(donors['domains'][domain]['cohort_labels'],device='cuda')
  for label,cohort in [(1,'continuation'),(2,'spaced_word')]:
   ids=torch.where((mapping>=0)&(cohorts==label))[0];h=hh[ids];delta=sources[mapping[ids]]-sources[ids];x0,x1=norm(h),norm(h+delta);t0,t1=teacher(x0),teacher(x1)
   centered_truth=(teacher(x1-mu)-teacher(x0-mu))@RU.T;original_truth=(t1-t0)@RU.T;full_truth=(delta+t1-t0)@RU.T
   row=dict(domain=domain,cohort=cohort,sites=len(ids),programs={})
   for name,p in programs.items():
    error=((student(x1,p)-student(x0,p))-(t1-t0))@RU.T
    row['programs'][name]=dict(centered_response_error=float(error.norm()/centered_truth.norm()),mlp_response_error=float(error.norm()/original_truth.norm()),full_residual_response_error=float(error.norm()/full_truth.norm()))
   records.append(row)
 pred=dict(pred_a_replay=max(checks)<1e-5,pred_b_transfer_gap=sum(r['programs']['learned']['centered_response_error']>2*.0326400716 for r in records)>=3,pred_c_ordering=sum(r['programs']['learned']['centered_response_error']<=r['programs']['fixed']['centered_response_error'] for r in records)>=3)
 result=dict(predictions=pred,records=records,maximum_replay=max(checks),seconds=time.monotonic()-start,scope='Exact exported-program response errors in original uncentered-vocabulary QR metric before finalRMS/softcap. Centered quadratic denominator matches training definition; other denominators expose metric dependence. Opened same-cohort native donor panel, not additional independent validation or causal attribution of the gap.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
