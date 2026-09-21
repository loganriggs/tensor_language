#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_localized_writer pred_c_predictive
"""Discovery profile of frozen stable groups1/2/3, reused16-document panel.
pred_a_instrument frozen hashes, finite effects, direct readout replay<1e-8.
pred_b_localized_writer some constituent has >25% centered vocabulary writer
energy in top256 coordinates. pred_c_predictive some group's removal adds>.005
nats/token. These are descriptive screens, no semantic/selective circuit claim.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=16,groups=['group1','group2','group3'],fit=False,reused_panel=True)));return
 import torch
 import torch.nn.functional as F
 import tiktoken
 from circuit_fast_screen_producer import Bilin18TorchBackend
 sys.path.insert(0,str(P));from native_feature_capture import capture
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'MIDPOINT_STABLE_GROUP_PROFILE_V1.json';assert not out.exists();start=time.perf_counter();enc=tiktoken.get_encoding('gpt2')
 decode=lambda i:enc.decode([int(i)]) if int(i)<enc.n_vocab else '<unused:'+str(i)+'>'
 plan=json.loads((P/'MIDPOINT_STABLE_GROUP_REMOVAL_PLAN_V1.json').read_text());artifact=P/'MIDPOINT_STABLE_GROUP_REMOVAL_PROGRAMS_V1.pt';assert hashlib.sha256(artifact.read_bytes()).hexdigest()==plan['program_sha256']
 tokens=torch.load(P/'MIDPOINT_STABLE_GROUP_REMOVAL_TOKENS_V1.pt',weights_only=True);assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==plan['token_sha256']
 e={k:v.cuda().double() for k,v in torch.load(artifact,weights_only=True)['half0'].items()}
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');invru=torch.linalg.inv(ru)
 residual_writers=invru@e['W'];vocab=model.lm_head.weight.double()@residual_writers;vocab-=vocab.mean(0)
 writer_profiles=[]
 for j in [1,4,2,3]:
  v=vocab[:,j];ix=v.abs().topk(256).indices;positive=v.topk(16).indices;negative=(-v).topk(16).indices
  writer_profiles.append(dict(component=j,top256_energy_fraction=float(v[ix].square().sum()/v.square().sum()),positive=[dict(token=int(i),text=decode(i),weight=float(v[i])) for i in positive],negative=[dict(token=int(i),text=decode(i),weight=float(v[i])) for i in negative]))
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 groups={k:plan['groups'][k] for k in ['group1','group2','group3']};rows=[];checks=[]
 def tokenclass(i):
  s=decode(i);z=s.strip()
  if '\n' in s:return 'newline'
  if not z:return 'whitespace'
  if z.isdigit():return 'digits'
  if all(not c.isalnum() for c in z):return 'punctuation'
  if z.isalpha():return 'letters'
  return 'mixed'
 for doc,row in enumerate(tokens):
  c=capture(model,row[None,:256].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=((h-m/2)/scale)[16:];m=(m/scale)[16:];state=c['final'].flatten(0,1)[16:];base=logits(state);target=row[17:257].cuda();ce=F.cross_entropy(base,target,reduction='none')
  left=n@e['A']-e['left_mean'];right=m@e['B']-e['right_mean'];phi=left*right
  for group,ids in groups.items():
   reduced=phi[:,ids]@e['W'][:,ids].T;delta=-reduced@invru.T
   if doc==0:checks.append(float((delta@ru.T+reduced).norm()/reduced.norm()))
   changed=logits(state+delta.float());effect=(changed-base).double();effect-=effect.mean(1,keepdim=True);loss=F.cross_entropy(changed,target,reduction='none')-ce
   linear=-phi[:,ids]@vocab[:,ids].T
   cosine=(effect*linear).sum(1)/(effect.norm(dim=1)*linear.norm(dim=1)).clamp_min(1e-30)
   energy=effect.square().sum(1)
   for j in range(240):
    pos=j+16;rows.append(dict(document=int(plan['documents'][doc]),position=pos,group=group,input_token=int(row[pos]),input_piece=decode(row[pos]),input_class=tokenclass(row[pos]),next_token=int(row[pos+1]),next_piece=decode(row[pos+1]),next_class=tokenclass(row[pos+1]),left=left[j,ids].cpu().tolist(),right=right[j,ids].cpu().tolist(),products=phi[j,ids].cpu().tolist(),effect_energy=float(energy[j]),ce_added=float(loss[j]),effect_to_linear_cosine=float(cosine[j]),base_next_logit=float(base[j,target[j]]),next_logit_change=float(effect[j,target[j]])))
 summary={};examples={}
 for group in groups:
  rr=[x for x in rows if x['group']==group];summary[group]=dict(ce_added=sum(x['ce_added'] for x in rr)/len(rr),median_effect_to_linear_cosine=float(torch.tensor([x['effect_to_linear_cosine'] for x in rr]).median()),token_classes={})
  for field in ['input_class','next_class']:
   summary[group]['token_classes'][field]={}
   for cl in sorted(set(x[field] for x in rr)):
    ar=[x for x in rr if x[field]==cl];summary[group]['token_classes'][field][cl]=dict(sites=len(ar),mean_effect_energy=sum(x['effect_energy'] for x in ar)/len(ar),ce_added=sum(x['ce_added'] for x in ar)/len(ar))
  chosen=[];docs=set()
  for x in sorted(rr,key=lambda x:x['effect_energy'],reverse=True):
   if x['document'] in docs:continue
   docs.add(x['document']);y=dict(x);row=tokens[x['document']-96];y['short_context']=' '.join(enc.decode(row[max(0,x['position']-16):x['position']+1].tolist()).split()[-6:]);chosen.append(y)
   if len(chosen)==6:break
  examples[group]=chosen
 predictions=dict(pred_a_instrument=max(checks)<1e-8 and all(torch.isfinite(torch.tensor(x['ce_added'])) for x in rows),pred_b_localized_writer=any(x['top256_energy_fraction']>.25 for x in writer_profiles),pred_c_predictive=any(x['ce_added']>.005 for x in summary.values()))
 result=dict(predictions=predictions,summary=summary,writer_profiles=writer_profiles,examples=examples,records=rows,readout_replay=max(checks),seconds=time.perf_counter()-start,scope='Descriptive hypothesis generation on reused16document removal panel. Frozen half0 groups; no fitting. Static writer signs refer to positive product coordinates, not guaranteed final-logit effects. No held-out semantic test or selectivity claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['records','examples','writer_profiles']},indent=2),flush=True)
if __name__=='__main__':main()
