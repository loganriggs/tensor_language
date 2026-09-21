#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_pairgeneralization pred_c_stategap
"""Frozen response dictionary, trained997 vsuntrained613/1379donors.
Replay<1e-5abs/cache<1e-4; unseen-cal<=1.5trainedcal; eval>=2calunseenshifts.
No fit;656products903168coeff each; polynomial metric only, no circuit adoption.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  import torch
  ids=torch.arange(2048)
  for shift in [997,613,1379]:assert math.gcd(shift,2048)==1 and (ids.roll(shift)!=ids).all() and ids.roll(shift).unique().numel()==2048
  print(json.dumps(dict(pair_permutations='pass',fit=False,programs=3,panels=2)));return
 import torch
 sys.path.insert(0,str(P));from quartic_finite_response import response_features
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'QUARTIC_PAIR_COVERAGE_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];xs=[r['rows'].cuda().double() for r in panels]
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight')/scale;L,R=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A,B=w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')
 def teacher(x):
  x=x.float();h=((x@A.T)*(x@B.T))@D.T;return (((h@L.T)*(h@R.T))@C.T).double()
 files={'half':'QUARTIC_RESPONSE_FEATURES_32_WEIGHT_1_V1.pt','value':'QUARTIC_RESPONSE_FEATURES_32_WEIGHT_0_V1.pt','path':'QUARTIC_PATH_FIT_V1.pt'};programs={}
 for name,f in files.items():
  p={k:v.cuda().double() for k,v in torch.load(P/f,weights_only=True).items()};p['C']=ru.double()@p['C']/scale;programs[name]=p
 rows=[];checks=[]
 for panel,x in enumerate(xs):
  y=teacher(x);old=data['targets'][panel].cuda().double();checks.append(float((y-old).norm()/old.norm()))
  for shift in [997,613,1379]:
   delta=.5*(x.roll(shift,0)-x);target=teacher(x+delta)-y;energy=target.square().sum();assert energy>0
   for name,p in programs.items():
    response=response_features(x,delta,p['U'],p['V'])@p['C'].T;row=dict(panel=panel,shift=shift,program=name,error=float((response-target).norm()/target.norm()),teacher_energy=float(energy));rows.append(row);print(json.dumps(row),flush=True)
 lookup={(r['panel'],r['shift'],r['program']):r['error'] for r in rows};replay=max(abs(lookup[i,997,'half']-v) for i,v in enumerate([.039233493810430585,.21113083561144408]))
 pred=dict(pred_a_integrity=replay<1e-5 and max(checks)<1e-4 and all(math.isfinite(r['error']) for r in rows),pred_b_pairgeneralization=all(lookup[0,s,'half']<=1.5*lookup[0,997,'half'] for s in [613,1379]),pred_c_stategap=all(lookup[1,s,'half']>=2*lookup[0,s,'half'] for s in [613,1379]))
 result=dict(predictions=pred,rows=rows,old_response_replay_absolute=replay,teacher_replay=max(checks),program_hashes={k:hashlib.sha256((P/f).read_bytes()).hexdigest() for k,f in files.items()},seconds=time.monotonic()-start,scope='Frozen opened-state pair diagnostic, unembedding polynomial numerator metric, no normalized native response/semantic/OOD adoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
