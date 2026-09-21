#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_absolute pred_c_relative
"""Frozen three-component graph, fresh identified-doc native confirmation.
48capturescontext256; individual1/2/3 pluscombined removal. pred_a final/source
replay<1e-5/1e-4 and exact donor constraints. pred_b natural/hybrid<=.15,
change<=.20 everydomain/cohort/selection. pred_c graph<=1.10separatebaseline
in everycell. Null: opened scalar fit hides OOD, donor, or composition failure.
512products vs768; both897804deduplicatedfloats. Nativez/h stillrequired.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def move(x,torch):
 if isinstance(x,dict):return {k:move(v,torch) for k,v in x.items()}
 return x.cuda().double() if isinstance(x,torch.Tensor) else x
def main():
 import torch
 sys.path.insert(0,str(P));from shared_mixed_source_graph import component_scalars
 from source_interface import residual_write as separate_write
 graph=torch.load(P/'PARTIAL_GRAPH_FROZEN_V1.pt',weights_only=True);baselines=torch.load(P/'MULTIMODE_PAIR_BASELINES_V1.pt',weights_only=True)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  torch.set_num_threads(2);z=torch.zeros(2,3,1152,dtype=torch.float64);h=torch.ones_like(z)
  assert component_scalars(z,h,graph).shape==(2,3,3)
  for b in baselines.values():assert separate_write(z,h,b).shape==(2,3,1152)
  print(json.dumps(dict(captures=48,context=256,selections=4,source_products=512,stored_floats=897804,shape_smoke='PASS')));return
 import torch.nn.functional as F
 import tiktoken
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from token_boundary_conditions import annotate
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/'PARTIAL_GRAPH_FRESH_NATIVE_V1.json';assert not out.exists()
 plan=json.loads((P/'PARTIAL_GRAPH_FRESH_PLAN_V1.json').read_text())
 for name,digest in plan['file_sha256'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 graph=move(graph,torch);baselines=move(baselines,torch);writer=graph['residual_writer'];fold=torch.load(P/'PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt',weights_only=True);Qs=fold['matrices'].cuda();A=fold['A'].cuda();B=fold['B'].cuda();alpha=fold['alpha'].cuda();beta=fold['beta'].cuda()
 tokens=torch.load(P/'PARTIAL_GRAPH_FRESH_TOKENS_V1.pt',weights_only=True);donors=torch.load(P/'PARTIAL_GRAPH_FRESH_DONORS_V1.pt',weights_only=True)
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];enc=tiktoken.get_encoding('gpt2');checks=[];qchecks=[];records=[]
 selections={'mode1':[0],'mode2':[1],'mode3':[2],'combined':[0,1,2]}
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 def native_components(h,m):
  h=h.double();m=m.double();s=(h.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[:,None]
  phi=((h@A-.5*(m@A))/s-alpha)*((m@B)/s-beta)
  return phi[:,:,None]*writer
 for domain,rows in tokens.items():
  assert hashlib.sha256(rows.numpy().tobytes()).hexdigest()==plan['tokens_sha256'][domain]
  cache={k:[] for k in ['z','h','m','final']};labels=[]
  for row in rows:
   c=capture(model,row[None,:256].cuda());z=c['x16'].flatten(0,1)[16:];h=c['h17'].flatten(0,1)[16:];m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).flatten(0,1)[16:];final=c['final'].flatten(0,1)[16:]
   replay=h+b17.mlp(F.rms_norm(h,(1152,)));checks.append(float((replay-final).double().norm()/final.double().norm()))
   true=torch.stack([m.double()@A,m.double()@B],-1).flatten(1);q=torch.einsum('ni,oij,nj->no',z.double(),Qs,z.double());qchecks.append(float((q-true).norm()/true.norm()))
   for k,v in [('z',z),('h',h),('m',m),('final',final)]:cache[k].append(v)
   labels+=annotate(row.tolist(),enc)[16:256]
  cache={k:torch.cat(v) for k,v in cache.items()};mapping=donors[domain];flat=rows[:,16:256].flatten();valid=(mapping>=0).nonzero().flatten();assert torch.all(flat[valid]==flat[mapping[valid]]) and torch.all(valid//240!=mapping[valid]//240)
  for offset in range(0,len(valid),96):
   ids=valid[offset:offset+96];ds=mapping[ids];gi=ids.cuda();gd=ds.cuda();h=cache['h'][gi];m=cache['m'][gi];md=cache['m'][gd];hybrid=h-m+md;states=[cache['final'][gi],hybrid+b17.mlp(F.rms_norm(hybrid,(1152,)))];targets=rows[:,17:257].flatten()[ids].cuda();bl=[logits(x) for x in states];ces=[F.cross_entropy(x,targets,reduction='none') for x in bl]
   zs=[cache['z'][gi].double(),cache['z'][gd].double()];hs=[h.double(),hybrid.double()]
   writes=dict(native=[native_components(h,m),native_components(hybrid,md)],separate=[torch.stack([separate_write(z,hh,baselines[str(j)]) for j in range(3)],1) for z,hh in zip(zs,hs)],graph=[component_scalars(z,hh,graph)[:,:,None]*writer for z,hh in zip(zs,hs)])
   for selection,components in selections.items():
    for name,ww in writes.items():
     effects=[];damages=[]
     for state,base,ce,w in zip(states,bl,ces,ww):
      changed=logits(state-w[:,components].sum(1).float());effect=(changed-base).double();effect-=effect.mean(-1,keepdim=True);effects.append(effect);damages.append(F.cross_entropy(changed,targets,reduction='none')-ce)
     families=dict(natural=effects[0],hybrid=effects[1],change=effects[1]-effects[0]);dc=dict(natural=damages[0],hybrid=damages[1],change=damages[1]-damages[0])
     if name=='native':ref=families;refce=dc
     for family,value in families.items():
      err=(value-ref[family]).square().sum(-1);energy=ref[family].square().sum(-1);ceerr=dc[family]-refce[family]
      for doc in (ids//240).unique().tolist():
       docmask=ids//240==doc
       for cohort in ['all','continuation','spaced_word']:
        condition=torch.ones(len(ids),dtype=torch.bool) if cohort=='all' else torch.tensor([labels[i][cohort] for i in ids.tolist()]);mask=(docmask&condition).cuda();n=int(mask.sum())
        if n:records.append(dict(domain=domain,document=plan['recipient_documents'][domain][doc],selection=selection,candidate=name,family=family,cohort=cohort,sites=n,error_energy=float(err[mask].sum()),reference_energy=float(energy[mask].sum()),ce_added_sum=float(dc[family][mask].sum()),ce_error_sum=float(ceerr[mask].sum())))
 summary={};cells=[]
 for domain in tokens:
  summary[domain]={}
  for selection in selections:
   summary[domain][selection]={}
   for name in ['native','separate','graph']:
    summary[domain][selection][name]={}
    for family in ['natural','hybrid','change']:
     summary[domain][selection][name][family]={}
     for cohort in ['all','continuation','spaced_word']:
      rr=[r for r in records if r['domain']==domain and r['selection']==selection and r['candidate']==name and r['family']==family and r['cohort']==cohort];n=sum(r['sites'] for r in rr);assert n>0;den=sum(r['reference_energy'] for r in rr);assert den>0
      summary[domain][selection][name][family][cohort]=dict(sites=n,effect_relative_error=(sum(r['error_energy'] for r in rr)/den)**.5,reference_logit_rms=(den/(n*bl[0].shape[-1]))**.5,ce_added=sum(r['ce_added_sum'] for r in rr)/n,ce_disagreement=sum(r['ce_error_sum'] for r in rr)/n)
   for family,limit in [('natural',.15),('hybrid',.15),('change',.20)]:
    for cohort in ['all','continuation','spaced_word']:
     g=summary[domain][selection]['graph'][family][cohort]['effect_relative_error'];b=summary[domain][selection]['separate'][family][cohort]['effect_relative_error'];cells.append(dict(domain=domain,selection=selection,family=family,cohort=cohort,graph_error=g,baseline_error=b,ratio=g/b,absolute_pass=g<=limit,relative_pass=g<=1.10*b))
 predictions=dict(pred_a_instrument=max(checks)<1e-5 and max(qchecks)<1e-4,pred_b_absolute=all(c['absolute_pass'] for c in cells),pred_c_relative=all(c['relative_pass'] for c in cells))
 result=dict(predictions=predictions,summary=summary,cells=cells,records=records,final_state_replay=max(checks),source_replay=max(qchecks),plan=plan,seconds=time.perf_counter()-start)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=predictions,failed_cells=[c for c in cells if not(c['absolute_pass'] and c['relative_pass'])],seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
