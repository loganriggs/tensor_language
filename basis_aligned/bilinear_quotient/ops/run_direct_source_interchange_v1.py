#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_hybrid pred_c_change
"""Same-token source interchange with fixed recipient background and native RMS.
pred_a_instrument finalstate replay<1e-5, original source scalar replay<1e-4,
exact donor token/doc constraints. pred_b_hybrid covariance16 leading-feature
removal error<=.15 every domain/cohort. pred_c_change change-in-removal error
<=.20 every domain/cohort. Null: ordinary-state correlation hides input-role failure.
48 reused captures, no fit. Linear4612,rank16 41508,rank64 152196 stored scalars;
source squares0/32/128 plus one final product; native z,h producers still required.
Arms native/isotropic_0/isotropic_16/covariance_16/covariance_64. Attention17 fixed;
not full upstream ablation. Target is leading-feature removal, not all source effects.
"""
import os,sys,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PROGRAM_FILE='MIDPOINT_SOURCE_INTERFACE_V1.pt'
PLAN_FILE='MIDPOINT_SOURCE_INTERCHANGE_PLAN_V1.json'
DONOR_FILE='MIDPOINT_SOURCE_INTERCHANGE_DONORS_V1.pt'
TOKEN_FILE='MIDPOINT_SOURCE_NATIVE_TOKENS_V1.pt'
OUTPUT_FILE='MIDPOINT_SOURCE_INTERCHANGE_V1.json'
PRIMARY='covariance_16'
ANCHOR='isotropic_0'
FORWARDS=48
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  import torch
  sys.path.insert(0,str(P));from source_interface import residual_write
  torch.set_num_threads(2);v=torch.load(P/PROGRAM_FILE,weights_only=True)
  for p in v.values():assert residual_write(torch.zeros(2,3,1152,dtype=torch.float64),torch.ones(2,3,1152,dtype=torch.float64),p).shape==(2,3,1152)
  print(json.dumps(dict(forwards=FORWARDS,context=256,fit=False,shape_smoke='PASS')));return
 import torch
 import torch.nn.functional as F
 import tiktoken
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from source_interface import residual_write
 from native_feature_capture import capture
 from token_boundary_conditions import annotate
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/OUTPUT_FILE;assert not out.exists()
 plan=json.loads((P/PLAN_FILE).read_text());assert hashlib.sha256((P/PROGRAM_FILE).read_bytes()).hexdigest()==plan['program_sha256'];assert hashlib.sha256((P/DONOR_FILE).read_bytes()).hexdigest()==plan['donor_sha256']
 programs={k:{n:t.cuda().double() for n,t in p.items()} for k,p in torch.load(P/PROGRAM_FILE,weights_only=True).items()};mode={k:t.cuda().double() for k,t in torch.load(P/'MIDPOINT_NATIVE_OBSERVER_MODES_V1.pt',weights_only=True).items()};a=mode['A'][:,0];b=mode['B'][:,0];e=programs[ANCHOR];writer=e['residual_writer'];fold=torch.load(P/'MIDPOINT_CONTINUATION_SOURCE_FOLD_V1.pt',weights_only=True);Qa=fold['a']['matrix'].cuda();Qb=fold['b']['matrix'].cuda()
 tokens=torch.load(P/TOKEN_FILE,weights_only=True);donors=torch.load(P/DONOR_FILE,weights_only=True);model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];enc=tiktoken.get_encoding('gpt2');checks=[];qchecks=[];records=[]
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 def native_write(h,m):
  h=h.double();m=m.double();s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();phi=((h@a-.5*(m@a))/s-e['alpha'])*((m@b)/s-e['beta']);return phi[:,None]*writer
 for domain,rows in tokens.items():
  assert hashlib.sha256(rows.numpy().tobytes()).hexdigest()==plan['tokens_sha256'][domain]
  cache={k:[] for k in ['z','h','m','final']};labels=[]
  for row in rows:
   c=capture(model,row[None,:256].cuda());z=c['x16'].flatten(0,1)[16:];h=c['h17'].flatten(0,1)[16:];m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).flatten(0,1)[16:];final=c['final'].flatten(0,1)[16:]
   replay=h+b17.mlp(F.rms_norm(h,(1152,)));checks.append(float((replay-final).double().norm()/final.double().norm()))
   true=torch.stack([m.double()@a,m.double()@b],1);q=torch.stack([((z.double()@Qa)*z.double()).sum(1),((z.double()@Qb)*z.double()).sum(1)],1);qchecks.append(float((q-true).norm()/true.norm()))
   for k,v in [('z',z),('h',h),('m',m),('final',final)]:cache[k].append(v)
   labels+=annotate(row.tolist(),enc)[16:256]
  cache={k:torch.cat(v) for k,v in cache.items()};mapping=donors[domain];flat=rows[:,16:256].flatten();valid=(mapping>=0).nonzero().flatten();assert torch.all(flat[valid]==flat[mapping[valid]]) and torch.all(valid//240!=mapping[valid]//240)
  for startidx in range(0,len(valid),96):
   ids=valid[startidx:startidx+96];ds=mapping[ids];gi=ids.cuda();gd=ds.cuda();h=cache['h'][gi];m=cache['m'][gi];md=cache['m'][gd];hybrid=h-m+md;states=[cache['final'][gi],hybrid+b17.mlp(F.rms_norm(hybrid,(1152,)))];targets=rows[:,17:257].flatten()[ids].cuda();base_logits=[logits(x) for x in states];ces=[F.cross_entropy(x,targets,reduction='none') for x in base_logits]
   writes={'native':[native_write(h,m),native_write(hybrid,md)],**{name:[residual_write(cache['z'][gi].double(),h.double(),p),residual_write(cache['z'][gd].double(),hybrid.double(),p)] for name,p in programs.items()}}
   for name,ww in writes.items():
    effects=[];damages=[]
    for state,base,ce,w in zip(states,base_logits,ces,ww):
     changed=logits(state-w.float());effect=(changed-base).double();effect-=effect.mean(-1,keepdim=True);effects.append(effect);damages.append(F.cross_entropy(changed,targets,reduction='none')-ce)
    families={'hybrid':effects[1],'change':effects[1]-effects[0]};dc={'hybrid':damages[1],'change':damages[1]-damages[0]}
    if name=='native':ref={k:v for k,v in families.items()};refce=dc
    for family,value in families.items():
     err=(value-ref[family]).square().sum(-1);energy=ref[family].square().sum(-1);ceerr=dc[family]-refce[family]
     for doc in (ids//240).unique().tolist():
      docmask=ids//240==doc
      for cohort in ['all','continuation','spaced_word']:
       condition=torch.ones(len(ids),dtype=torch.bool) if cohort=='all' else torch.tensor([labels[i][cohort] for i in ids.tolist()]);mask=(docmask&condition).cuda();n=int(mask.sum())
       if not n:continue
       records.append(dict(domain=domain,document=plan['recipient_documents'][domain][doc],candidate=name,family=family,cohort=cohort,sites=n,error_energy=float(err[mask].sum()),reference_energy=float(energy[mask].sum()),ce_added_sum=float(dc[family][mask].sum()),ce_error_sum=float(ceerr[mask].sum())))
 summary={}
 for domain in tokens:
  summary[domain]={}
  for name in writes:
   summary[domain][name]={}
   for family in ['hybrid','change']:
    summary[domain][name][family]={}
    for cohort in ['all','continuation','spaced_word']:
     rr=[r for r in records if r['domain']==domain and r['candidate']==name and r['family']==family and r['cohort']==cohort];n=sum(r['sites'] for r in rr);assert n>0;den=sum(r['reference_energy'] for r in rr)
     summary[domain][name][family][cohort]=dict(sites=n,effect_relative_error=(sum(r['error_energy'] for r in rr)/den)**.5,reference_logit_rms=(den/(n*base_logits[0].shape[-1]))**.5,ce_added=sum(r['ce_added_sum'] for r in rr)/n,ce_disagreement=sum(r['ce_error_sum'] for r in rr)/n)
 pred=dict(pred_a_instrument=max(checks)<1e-5 and max(qchecks)<1e-4,pred_b_hybrid=all(summary[d][PRIMARY]['hybrid'][c]['effect_relative_error']<=.15 for d in tokens for c in ['all','continuation','spaced_word']),pred_c_change=all(summary[d][PRIMARY]['change'][c]['effect_relative_error']<=.20 for d in tokens for c in ['all','continuation','spaced_word']))
 result=dict(predictions=pred,summary=summary,records=records,final_state_replay=max(checks),source_replay=max(qchecks),plan=plan,seconds=time.perf_counter()-start);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
if __name__=='__main__':main()
