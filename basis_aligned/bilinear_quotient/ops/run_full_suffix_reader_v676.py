#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_endpoint pred_b_reader_closure pred_c_causal_reader
"""Full dynamic suffix finite-reader instrument, not a predictive model.

48 opened rows,8 prefix+16 suffix calls,8 analytical reverse passes with4readers
(48 attention pullbacks,96 reference attention forwards). No fitting/autograd.
Null: normalized attention cross-position terms break recursive closure.
"""
import os,sys,json,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];POLY=ROOT/'basis_aligned/polynomial_causal'
OUT=ROOT/'basis_aligned/bilinear_quotient/circuits/followups/full_suffix_reader_v676_result.json'
PREDICTIONS=dict(pred_a_native_endpoint='native graph margin replay<=1e-4',
 pred_b_reader_closure='all four readers at all seven boundaries absolute/norm-scaled closure<=1e-4',
 pred_c_causal_reader='early-token reader has exactly zero coefficients on future tokens at every boundary')

def main():
 plan=dict(rows=48,prefix_calls=8,suffix_calls=16,analytic_reverse_passes=8,readers_per_pass=4,
           attention_pullbacks=48,reference_attention_forwards=96,optimizer_steps=0,autograd_backwards=0,
           execution_policy='managed_queue_only',predictions=PREDICTIONS)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(plan));return
 import torch
 import torch.nn.functional as F
 import tiktoken
 import circuit_fast_screen_producer as producer
 import subject_number_sparse_graph_token_extraction_v1 as graph
 from disk_guard import guard_write
 sys.path.insert(0,str(POLY))
 from full_suffix_readers import secant_readers
 if OUT.exists():raise FileExistsError(OUT)
 torch.set_grad_enabled(False);torch.set_num_threads(4);tic=time.perf_counter()
 model=producer.Bilin18TorchBackend.load('cuda').model;eps=torch.finfo(torch.float32).eps
 decoder=json.loads((POLY/'SUBJECT_NUMBER_EMBEDDING_DECODER_FROZEN_V1_ARTIFACT.json').read_text())['frozen_decoder']
 axis=torch.tensor(decoder['axis'],device='cuda',dtype=torch.float64);unit=axis/axis.norm();threshold=float(decoder['threshold'])/float(axis.norm())
 encoding=tiktoken.get_encoding('gpt2');pairs=[['is','are'],['can','will'],['may','might'],['should','could']]
 token_pairs=[[encoding.encode(' '+word) for word in pair] for pair in pairs]
 assert all(len(ids)==1 for pair in token_pairs for ids in pair)
 answer_pairs=torch.tensor([[ids[0] for ids in pair] for pair in token_pairs],device='cuda')
 U=model.lm_head.weight[answer_pairs].double()
 rows=json.loads((POLY/'SUBJECT_POSITION_STRESS_V644_ROWS.json').read_text())
 counts=dict(prefix_calls=0,suffix_calls=0,analytic_reverse_passes=0)
 endpoint=[];absolute=[];scaled=[];causal=[];records=[]
 def trace(raw,x0,first):
  counts['suffix_calls']+=1;x=raw
  result=dict(states=[],raw_attention_inputs=[],mlp_inputs=[])
  for layer in range(11,18):
   block=model.transformer.h[layer]
   if layer>11:x=block.lambdas[0]*x+block.lambdas[1]*x0
   if layer>=12:result['raw_attention_inputs'].append(x.double())
   a,first=block.attn(F.rms_norm(x,(x.shape[-1],)),first);h=x+a
   if layer>=12:result['mlp_inputs'].append(h.double())
   x=h+block.mlp(F.rms_norm(h,(h.shape[-1],)));result['states'].append(x.double())
  return result
 def margins(x,pos):
  selected=x[torch.arange(len(x),device=x.device),pos]
  raw=torch.einsum('bd,rcd->rbc',F.rms_norm(selected,(selected.shape[-1],),eps=eps),U)
  logits=30*torch.tanh(raw/30)
  return logits[...,0]-logits[...,1]
 for template in dict.fromkeys(r['template'] for r in rows):
  entries=[r for r in rows if r['template']==template]
  ids=torch.tensor([r['token_ids'] for r in entries],device='cuda');pos=torch.tensor([r['subject_position'] for r in entries],device='cuda')
  readpos=torch.tensor([r['readout_position'] for r in entries],device='cuda');batch=torch.arange(len(entries),device='cuda')
  initial=F.rms_norm(model.transformer.wte(ids),(model.config.n_embd,)).float()
  subject=initial[batch,pos].double();orth=subject-(subject@unit)[:,None]*unit
  scale=((subject.square().sum(-1)-threshold**2)/orth.square().sum(-1)).sqrt()
  removed=initial.clone();removed[batch,pos]=(threshold*unit+scale[:,None]*orth).float()
  raw,x0,first,pb,_=graph._capture(model,initial,torch,F);_,_,_,pr,_=graph._capture(model,removed,torch,F);counts['prefix_calls']+=2
  edited=raw.clone();edited[batch,pos]+=sum((r-b)[batch,pos] for b,r in zip(pb,pr)).float()
  base,changed=trace(raw,x0,first),trace(edited,x0,first)
  for source,tr in [(raw,base),(edited,changed)]:
   target=graph._suffix_margin(model,source,x0,first,readpos,answer_pairs[0].expand(len(entries),-1),torch,F);counts['suffix_calls']+=1
   endpoint.append(float((margins(tr['states'][-1],readpos)[0]-target).abs().max()))
  blocks=[];b,t,d=raw.shape
  for module in model.transformer.h[12:18]:
   a=module.attn;cos,sin=a.rotary(torch.zeros(b,t,a.n_head,a.head_dim,device='cuda'))
   blocks.append(dict(left=module.mlp.Left.weight.double(),right=module.mlp.Right.weight.double(),down=module.mlp.Down.weight.double(),
    lambdas=module.lambdas.double(),attention={key:getattr(a,name).weight.double() for key,name in [('q','c_q'),('k','c_k'),('q2','c_q2'),('k2','c_k2'),('v','c_v'),('o','c_proj')]},
    mixture=a.lamb.double(),heads=a.n_head,head_eps=eps,cos=cos[0,:,0].double(),sin=sin[0,:,0].double()))
  readers=secant_readers(blocks,base,changed,first.reshape(b,t,9,128).double(),U,readpos,eps);counts['analytic_reverse_passes']+=1
  target=margins(changed['states'][-1],readpos)-margins(base['states'][-1],readpos)
  for l,q in enumerate(readers):
   delta=changed['states'][l]-base['states'][l];got=(q*delta).sum((-1,-2));error=got-target
   norm=q.square().sum((-1,-2)).sqrt()*delta.square().sum((-1,-2)).sqrt()+target.abs()
   absolute.append(float(error.abs().max()));scaled.append(float((error.abs()/norm.clamp_min(1e-20)).max()))
  early=torch.ones_like(readpos)
  earlier=secant_readers(blocks,base,changed,first.reshape(b,t,9,128).double(),U,early,eps);counts['analytic_reverse_passes']+=1
  causal.extend(float(q[:,:,2:].abs().max()) for q in earlier)
  records.append(dict(template=template,max_boundary_absolute=max(absolute[-7:]),max_boundary_scaled=max(scaled[-7:])))
  print(template,records[-1],flush=True)
 a=max(endpoint)<=1e-4 and counts==dict(prefix_calls=8,suffix_calls=16,analytic_reverse_passes=8)
 b=a and max(absolute+scaled)<=1e-4;c=a and max(causal)==0
 result=dict(created_utc=datetime.now(timezone.utc).isoformat(),plan=plan,counts=counts,predictions=dict(zip(PREDICTIONS,[bool(a),bool(b),bool(c)])),
  max_endpoint_error=max(endpoint),max_boundary_absolute_error=max(absolute),max_boundary_scaled_error=max(scaled),max_causal_leakage=max(causal),records=records,
  wall_seconds=time.perf_counter()-tic,scope='Full dynamic two-endpoint finite readers; requires edited executions, no predictive or low-rank guarantee')
 payload=json.dumps(result,indent=2)+'\n';guard_write(len(payload.encode()),str(OUT.parent),'full suffix reader');OUT.write_text(payload)
 print(json.dumps(result['predictions']));assert a and b and c
if __name__=='__main__':main()
