#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_moments pred_b_ranking pred_c_global
"""2048 weight-generated MLP16 sources; carry/source RMS ratios .5,1,2.
Frozen3686product/14067072coefficient replacements, no fitting or model forwards.
Moments: trace relative error<.1, mean error/source RMSnorm<.1, finite errors.
Mixed response error<unguarded learned at all ratios; isotropic<pruned at all.
Null: artificial law does not explain native response ranking. Explicit RMS and
softcap. Carry independent of source; synthetic outcomes are not native evidence.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,samples=2048,ratios=[.5,1.,2.],batch=128)));return
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from spherical_source_moments import moments
 start=time.monotonic();out=P/'SPHERICAL_SOURCE_GEOMETRY_V1.json';assert not out.exists()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 L,R,D=[w('transformer.h.16.mlp.'+k+'.weight') for k in ['Left','Right','Down']];scale=w('transformer.h.17.lambdas')[0];bias16=scale*w('transformer.h.16.mlp.Down_bias')
 mean,cov=moments(L.double(),R.double(),(scale*D).double());second=cov.trace()+mean.square().sum();source_rms=(second/1152).sqrt().float()
 torch.manual_seed(180606);torch.cuda.manual_seed_all(180606)
 def sphere(n):
  z=torch.randn(n,1152,device='cuda');return z/z.square().mean(-1,keepdim=True).sqrt()
 z=sphere(2048);source=((z@L.T)*(z@R.T))@(scale*D).T;carry=sphere(2048);donor=source.roll(1,0)
 trace_error=float((source.double().square().sum(1).mean()/second-1).abs());mean_error=float((source.double().mean(0)-mean).norm()/second.sqrt())
 del L,R,D,z,cov
 L,R,D=[w('transformer.h.17.mlp.'+k+'.weight') for k in ['Left','Right','Down']];bias17=w('transformer.h.17.mlp.Down_bias');U=w('lm_head.weight');eps=torch.finfo(torch.float32).eps
 paths=dict(pruned='FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt',learned='LEARNED_FULL_QUADRATIC_PROGRAM_V1.pt',mixed='FULL_QUADRATIC_MIXED_PROGRAM_V1.pt',isotropic='FULL_QUADRATIC_ISOTROPIC_PROGRAM_V1.pt')
 programs={name:{k:v.cuda().float() for k,v in torch.load(P/path,weights_only=True).items()} for name,path in paths.items()}
 def norm(x):return x/(x.square().mean(-1,keepdim=True)+eps).sqrt()
 def logits(x):
  y=30*torch.tanh((norm(x)@U.T)/30);return y-y.mean(-1,keepdim=True)
 def mlp(x,p=None):
  x=norm(x)
  if p is None:return ((x@L.T)*(x@R.T))@D.T+bias17
  return ((x@p['a'].T)*(x@p['b'].T))@p['writer'].T+x@p['linear'].T+p['bias']+bias17
 records=[]
 with torch.no_grad():
  for ratio in [.5,1.,2.]:
   sums={name:[0.,0.] for name in programs};natural_den=0.;response_den=0.
   for lo in range(0,2048,128):
    c=ratio*source_rms*carry[lo:lo+128]+bias16;h=c+source[lo:lo+128];hd=c+donor[lo:lo+128]
    baseline=logits(h);teacher=logits(h+mlp(h));teacher_d=logits(hd+mlp(hd));response=teacher_d-teacher
    natural_den+=float((teacher-baseline).double().square().sum());response_den+=float(response.double().square().sum())
    for name,p in programs.items():
     student=logits(h+mlp(h,p));student_d=logits(hd+mlp(hd,p));sums[name][0]+=float((student-teacher).double().square().sum());sums[name][1]+=float((student_d-student-response).double().square().sum())
   record=dict(carry_source_ratio=ratio,natural_denominator_squared=natural_den,response_denominator_squared=response_den,programs={name:dict(natural_error=(a/natural_den)**.5,response_error=(b/response_den)**.5) for name,(a,b) in sums.items()});records.append(record);print(json.dumps(record),flush=True)
 finite=all(torch.isfinite(torch.tensor([p['natural_error'],p['response_error']])).all().item() for row in records for p in row['programs'].values())
 pred=dict(pred_a_moments=trace_error<.1 and mean_error<.1 and finite,pred_b_ranking=all(r['programs']['mixed']['response_error']<r['programs']['learned']['response_error'] for r in records),pred_c_global=all(r['programs']['isotropic']['response_error']<r['programs']['pruned']['response_error'] for r in records))
 result=dict(predictions=pred,source_rms=float(source_rms),source_second_moment_trace_relative_error=trace_error,source_mean_normalized_error=mean_error,records=records,seconds=time.monotonic()-start,scope='Artificial independent carry and weight-generated sphere source. Frozen data-informed candidates; no new fit. Full-vocabulary centered logit errors with explicit final RMS/softcap; source donor cyclic shift. No native distribution or semantic interpretation established.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
