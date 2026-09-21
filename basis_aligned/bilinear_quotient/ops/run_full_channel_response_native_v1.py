#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_full pred_c_mode
"""Frozen primary response-refit: direct source swaps with recomputed RMS17.
Native final-state replay<1e-5; full path and mode-mediated effect errors<.10
in every primary same-cohort domain/cohort cell. Opposite cohort secondary.
Attention17 held recipient: direct residual path, not all MLP16 descendants.
Preregisteredlambda1 response-refit program3686products/14,067,072coefficients, no fit,48captures.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(captures=48,fit=False,cohorts=['all','continuation','spaced_word'])));return
 import torch,tiktoken
 import torch.nn.functional as F
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 from token_boundary_conditions import annotate
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic()
 out=P/'FULL_CHANNEL_RESPONSE_NATIVE_V1.json';assert not out.exists()
 e={k:v.cuda().double() for k,v in torch.load(P/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True).items()};p={k:v.cuda() for k,v in torch.load(P/'FULL_CHANNEL_RESPONSE_PROGRAM_V1.pt',weights_only=True).items()}
 a,b=p['a'].double(),p['b'].double();c=e['q']@e['R_U']@p['writer'].double();student=a.T@(c[:,None]*b)+b.T@(c[:,None]*a);native=e['native_matrix'];rw=torch.linalg.solve(e['R_U'],e['writer'])
 cal=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);roots=[];inverses=[]
 for key,mean in [('n','mean_n'),('m','mean_m')]:
  x=cal[key].flatten(0,1).cuda().double()-e[mean];ev,V=torch.linalg.eigh(x.T@x/len(x));mask=ev>1e-10*ev.max();roots.append((V*(ev.clamp_min(0)*mask).sqrt())@V.T);inverses.append((V*torch.where(mask,ev.clamp_min(1e-30).rsqrt(),0))@V.T)
 matrices={'native':native,'student':student};modes={}
 for name,matrix in matrices.items():
  U,s,Vh=torch.linalg.svd(roots[0]@matrix@roots[1],full_matrices=False);modes[name]=(inverses[0]@U[:,0]*s[0].sqrt(),inverses[1]@Vh[0]*s[0].sqrt())

 tokens=torch.load(P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt',weights_only=True);donors=json.loads((P/'FULL_CHANNEL_CONTINUATION_INTERCHANGE_DONORS_V1.json').read_text());assert hashlib.sha256((P/'FULL_CHANNEL_FRESH_TOKENS_V1.pt').read_bytes()).hexdigest()==donors['token_sha256']
 model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];records=[];checks=[];scale_rows=[]
 def logits(x):return 30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 def endpoint(h,name):
  z=F.rms_norm(h,(1152,))
  if name=='native':return h+b17.mlp(z)
  return h+((z@p['a'].T)*(z@p['b'].T))@p['writer'].T+z@p['linear'].T+p['bias']+b17.mlp.Down_bias
 def phi(h,source,name):
  h,source=h.double(),source.double();scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-source/2)/scale-e['mean_n'];m=source/scale-e['mean_m'];return (n@modes[name][0])*(m@modes[name][1])
 for domain,documents in tokens.items():
  hh=[];sources=[]
  for row in documents:
   cache=capture(model,row[None,:256].cuda());h=cache['h17'];checks.append(float((endpoint(h,'native')-cache['final']).norm()/cache['final'].norm()));hh.append(h.flatten(0,1)[16:255]);sources.append((b17.lambdas[0]*(cache['m16']-b16.mlp.Down_bias)).flatten(0,1)[16:255])
  hh=torch.cat(hh);sources=torch.cat(sources);cohorts=torch.tensor(donors['domains'][domain]['cohort_labels'],device='cuda');targets=documents[:,17:256].flatten().cuda()
  for family,mapping in donors['domains'][domain]['maps'].items():
   mapping=torch.tensor(mapping,device='cuda')
   for doc in range(len(documents)):
    ids=torch.arange(doc*239,(doc+1)*239,device='cuda');ids=ids[mapping[ids]>=0]
    if not len(ids):continue
    dst=mapping[ids];h=hh[ids];source=sources[ids];other=sources[dst];newh=h+(other-source);ratio=(newh.square().mean(-1)/h.square().mean(-1)).sqrt();scale_rows.append(dict(domain=domain,family=family,document_index=doc,rms_ratio_min=float(ratio.min()),rms_ratio_max=float(ratio.max())))
    effects={};damage={}
    for name in ('native','student'):
     state=endpoint(h,name);base=logits(state);basece=F.cross_entropy(base,targets[ids],reduction='none');delta=phi(newh,other,name)-phi(h,source,name)
     for kind,changedstate in [('full',endpoint(newh,name)),('mode',state+(delta[:,None]*rw).float())]:
      changed=logits(changedstate);effect=(changed-base).double();effect-=effect.mean(1,keepdim=True);effects[name,kind]=effect;damage[name,kind]=F.cross_entropy(changed,targets[ids],reduction='none')-basece
    for kind in ('full','mode'):
     for cohort,label in [('continuation',1),('spaced_word',2)]:
      mask=cohorts[ids]==label
      if not mask.any():continue
      records.append(dict(domain=domain,family=family,document_index=doc,kind=kind,cohort=cohort,sites=int(mask.sum()),reference_energy=float(effects['native',kind][mask].square().sum()),error_energy=float((effects['student',kind][mask]-effects['native',kind][mask]).square().sum()),native_ce_sum=float(damage['native',kind][mask].sum()),student_ce_sum=float(damage['student',kind][mask].sum())))
 keys=('domain','family','kind','cohort');summary=[]
 for values in sorted(set(tuple(r[k] for k in keys) for r in records)):
  rr=[r for r in records if tuple(r[k] for k in keys)==values];sites=sum(r['sites'] for r in rr);energy=sum(r['reference_energy'] for r in rr)
  summary.append(dict(zip(keys,values),sites=sites,effect_error=(sum(r['error_energy'] for r in rr)/energy)**.5 if energy else None,native_effect_rms=(energy/(sites*50304))**.5,native_ce_added=sum(r['native_ce_sum'] for r in rr)/sites,student_ce_added=sum(r['student_ce_sum'] for r in rr)/sites))
 primary=[r for r in summary if r['family']=='same_cohort'];pred=dict(pred_a_replay=max(checks)<1e-5,pred_b_full=all(r['effect_error'] is not None and r['effect_error']<.1 for r in primary if r['kind']=='full'),pred_c_mode=all(r['effect_error'] is not None and r['effect_error']<.1 for r in primary if r['kind']=='mode'))
 result=dict(predictions=pred,summary=summary,records=records,rms_ratios=scale_rows,maximum_replay=max(checks),seconds=time.monotonic()-start,scope='Path-specific MLP16 polynomial residual swap, recipient attention17 and other carry held fixed; h17/RMS17/MLP17/final RMS/softcap recomputed. Full effect compares true direct-path endpoints. Mode effect separately isolates scalar mode delta in each original background, not a claim to explain the whole path effect. Opened token-matched panel, no new fit/fresh validation. Upstream source extraction still external.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('records','rms_ratios')}),flush=True)
if __name__=='__main__':main()
