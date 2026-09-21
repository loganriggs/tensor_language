#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_absolute pred_c_relative
"""Opened two-panel scalar screen of consistent residual generation; no fitting.
Native h vector replaced by recipient residual projection plus source read.
Retain supplied RMS17 scale and native attention/embedding projected context.
Do not claim token extraction, independent fresh evidence, or endpoint fidelity.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

COHORTS=tuple(c+suffix for c in ('all','continuation','spaced_word') for suffix in ('','@early','@middle','@late'))

def main():
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);sys.path.insert(0,str(P))
 from generated_residual_interface import components,control
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(control());return
 from pairwise_component_interface import component_scalars,move_preserving_dtype
 from pairwise_reader_graph import source_reads
 from read_error_terms import baseline_reads
 from named_carry_error_terms import capture
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from token_boundary_conditions import annotate
 import tiktoken
 plan=json.loads((P/'POSITION_TRANSFER_PLAN_V1.json').read_text())
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 start=time.monotonic();torch.backends.cuda.matmul.allow_tf32=False
 model=Bilin18TorchBackend.load('cuda').model.float();graph=move_preserving_dtype(torch.load(P/'INTERCHANGE_READ_PAIRED_V2.pt',weights_only=True),'cuda');bundles={name:move_preserving_dtype(torch.load(P/file,weights_only=True),'cuda') for name,file in [('separate','FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt'),('isotropic','FRONTIER_FRESH_BASELINE_NATIVE_ISOTROPIC_V1.pt')]}
 cartesian_graph=move_preserving_dtype(torch.load(P/'INTERCHANGE_READ_CARTESIAN_V2.pt',weights_only=True),'cuda');fold=move_preserving_dtype(torch.load(P/'PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt',weights_only=True),'cuda');A,B,alpha,beta=[fold[k] for k in ('A','B','alpha','beta')];enc=tiktoken.get_encoding('gpt2');checks=[];state_checks=[];records=[];prefix_checks=[]
 for panel in (1,2):
  tokens=torch.load(P/f'FRONTIER_FRESH_TOKENS_V{panel}.pt',weights_only=True);maps=torch.load(P/f'FRONTIER_FRESH_DONORS_V{panel}.pt',weights_only=True)
  for domain in ('fineweb','stdlib'):
   cache={k:[] for k in ('z','h','m','carry')};labels=[]
   for row_index,row in enumerate(tokens[domain]):
    raw=capture(model,row[None,:256].cuda())
    if row_index==0:
     short=capture(model,row[None,:64].cuda());prefix_checks.append(max(float((raw[k][:,:64].double()-short[k].double()).norm()/short[k].double().norm()) for k in ('x16','h17')));assert prefix_checks[-1]<1e-4
    z=raw['x16'].flatten(0,1)[16:].double();h=raw['h17'].flatten(0,1)[16:];m=(model.transformer.h[17].lambdas[0]*(raw['m16']-model.transformer.h[16].mlp.Down_bias)).flatten(0,1)[16:]
    r=(raw['incoming17'].double()-raw['m16'].double()).flatten(0,1)[16:];s16=(r.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();lam=model.transformer.h[17].lambdas.double();gamma=(lam[1]*raw['x0'].double()+raw['attn17'].double()+lam[0]*model.transformer.h[16].mlp.Down_bias.double()).flatten(0,1)[16:]@A
    carry=lam[0]*s16[:,None]*(z@A)+gamma
    state_checks.append(float((s16[:,None]*z-r).norm()/r.norm()))
    for k,v in [('z',z),('h',h),('m',m),('carry',carry)]:cache[k].append(v)
    labels+=annotate(row.tolist(),enc)[16:256]
   cache={k:torch.cat(v) for k,v in cache.items()};mapping=maps[domain];valid=(mapping>=0).nonzero().flatten()
   for offset in range(0,len(valid),96):
    ids=valid[offset:offset+96];donors=mapping[ids];gi,gd=ids.cuda(),donors.cuda();h,m,md=cache['h'][gi],cache['m'][gi],cache['m'][gd];hybrid=h-m+md;vv={k:[] for k in ['true']+[f'{c}_{mode}' for c in ('graph','cartesian','separate','isotropic') for mode in ('boundary','generated')]}
    for z,hh,mm in [(cache['z'][gi],h,m),(cache['z'][gd],hybrid,md)]:
     hh,mm=hh.double(),mm.double();s=(hh.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt();truth_reads=torch.stack((mm@A,mm@B),-1).flatten(-2);native=((hh@A-.5*truth_reads[...,::2])/s[:,None]-alpha)*(truth_reads[...,1::2]/s[:,None]-beta);vv['true'].append(native)
     exact=components(truth_reads,cache['carry'][gi],s,alpha,beta);checks.append(float((exact-native).norm()/native.norm().clamp_min(1e-30)))
     fitted={'graph':source_reads(z,graph),'cartesian':source_reads(z,cartesian_graph)}
     for name,bundle in bundles.items():fitted[name]=torch.cat([baseline_reads(z,bundle[str(j)]) for j in range(3)],-1)
     for name,reads in fitted.items():
      vv[name+'_boundary'].append(((hh@A-.5*reads[...,::2])/s[:,None]-alpha)*(reads[...,1::2]/s[:,None]-beta));vv[name+'_generated'].append(components(reads,cache['carry'][gi],s,alpha,beta))
    for family,index in [('natural',0),('hybrid',1),('change',None)]:
     values={k:(v[index] if index is not None else v[1]-v[0]) for k,v in vv.items()};values={k:torch.cat((v,v.sum(-1,keepdim=True)),-1) for k,v in values.items()}
     for cohort in COHORTS:
      base_cohort=cohort.split('@')[0];mask=torch.ones(len(ids),dtype=torch.bool) if base_cohort=='all' else torch.tensor([labels[i][base_cohort] for i in ids.tolist()])
      if '@' in cohort:
       low,high={'early':(16,64),'middle':(64,128),'late':(128,256)}[cohort.split('@')[1]];positions=ids%240+16;mask &= (positions>=low)&(positions<high)
      mask=mask.cuda();n=int(mask.sum())
      if not n:continue
      truth=values['true'][mask]
      for name,v in values.items():
       if name=='true':continue
       records.append(dict(panel=panel,domain=domain,family=family,cohort=cohort,candidate=name,n=n,error_energy=(v[mask]-truth).square().sum(0).tolist(),truth_energy=truth.square().sum(0).tolist(),truth_sum=truth.sum(0).tolist()))
 summaries=[]
 for panel in (1,2):
  for domain in ('fineweb','stdlib'):
   for family in ('natural','hybrid','change'):
    for cohort in COHORTS:
     cell=dict(panel=panel,domain=domain,family=family,cohort=cohort,candidates={})
     for candidate in vv:
      if candidate=='true':continue
      rr=[r for r in records if all(r[k]==cell[k] for k in ('panel','domain','family','cohort')) and r['candidate']==candidate];n=sum(r['n'] for r in rr);energy=torch.tensor([r['error_energy'] for r in rr],dtype=torch.float64).sum(0);truth2=torch.tensor([r['truth_energy'] for r in rr],dtype=torch.float64).sum(0);truth1=torch.tensor([r['truth_sum'] for r in rr],dtype=torch.float64).sum(0)
      cell['candidates'][candidate]=dict(relative_error=(energy/(truth2-truth1.square()/n)).sqrt().tolist(),error_energy=energy.tolist())
     summaries.append(cell)
 absolute=all(max(c['candidates'][candidate+'_generated']['relative_error'])<=(.20 if c['family']=='change' else .15) for c in summaries for candidate in ('graph','cartesian'))
 relative=all(all(g<=1.1*b for g,b in zip(c['candidates'][candidate+'_generated']['relative_error'],c['candidates'][base+'_generated']['relative_error'])) for c in summaries for base in ('separate','isotropic') for candidate in ('graph','cartesian'))
 pred=dict(pred_a_replay=max(checks+state_checks)<1e-5,pred_b_absolute=absolute,pred_c_relative=relative)
 out=dict(prefix_replay_checks=prefix_checks,predictions=pred,native_component_replay=max(checks),residual16_replay=max(state_checks),summaries=summaries,seconds=time.monotonic()-start,plan=plan,scope='Opened-panel scalar screen only. RMS17 scale and attention/embedding context remain native; not a full-model replacement. Donor source changes while recipient carry stays fixed.')
 (P/'POSITION_TRANSFER_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(predictions=pred,replay=max(checks),seconds=out['seconds'])),flush=True)
if __name__=='__main__':main()
