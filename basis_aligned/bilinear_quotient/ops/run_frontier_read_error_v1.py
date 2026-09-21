#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_second_read pred_c_frozen
"""Exact signed three-term source error decomposition on opened code only.
Frozen graph, baseline and donors; no fitting. Predictions in the hashed-input plan.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 sys.path.insert(0,str(P));from pairwise_component_interface import component_scalars,move_preserving_dtype
 from source_interface import residual_write
 from read_error_terms import split,baseline_reads,control
 from pairwise_reader_graph import source_reads
 plan=json.loads((P/'FRONTIER_READ_ERROR_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(control());torch.set_num_threads(2);graph=torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True);z=torch.ones(2,3,1152,dtype=torch.float64);assert component_scalars(z,z,graph).shape==(2,3,3);print('32captures, same frozen graph; batched scalar preflight PASS');return
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 import torch.nn.functional as F
 import tiktoken
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from token_boundary_conditions import annotate
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();model=Bilin18TorchBackend.load('cuda').model.float();graph=move_preserving_dtype(torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True),'cuda');baseline=move_preserving_dtype(torch.load(P/'FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt',weights_only=True),'cuda');fold=move_preserving_dtype(torch.load(P/'PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt',weights_only=True),'cuda');A,B=fold['A'],fold['B'];alpha,beta=fold['alpha'],fold['beta'];writer=graph['residual_writer'];checks=[];records=[];term_checks=[];enc=tiktoken.get_encoding('gpt2')
 for version in (1,2):
  tokens=torch.load(P/f'FRONTIER_FRESH_TOKENS_V{version}.pt',weights_only=True)['stdlib'];mapping=torch.load(P/f'FRONTIER_FRESH_DONORS_V{version}.pt',weights_only=True)['stdlib'];documents=json.loads((P/f'FRONTIER_FRESH_PLAN_V{version}.json').read_text())['recipient_documents']['stdlib'];cache={k:[] for k in ('z','h','m')};labels=[]
  for row in tokens:
   raw=capture(model,row[None,:256].cuda());z=raw['x16'].flatten(0,1)[16:];h=raw['h17'].flatten(0,1)[16:];m=(model.transformer.h[17].lambdas[0]*(raw['m16']-model.transformer.h[16].mlp.Down_bias)).flatten(0,1)[16:]
   q=torch.einsum('ni,oij,nj->no',z.double(),fold['matrices'],z.double());true=torch.stack([m.double()@A,m.double()@B],-1).flatten(1);checks.append(float((q-true).norm()/true.norm()))
   for key,value in (('z',z),('h',h),('m',m)):cache[key].append(value)
   labels+=annotate(row.tolist(),enc)[16:256]
  cache={k:torch.cat(v) for k,v in cache.items()};valid=(mapping>=0).nonzero().flatten();flat=tokens[:,16:256].flatten();assert torch.all(flat[valid]==flat[mapping[valid]]) and torch.all(valid//240!=mapping[valid]//240)
  for offset in range(0,len(valid),96):
   ids=valid[offset:offset+96];donors=mapping[ids];gi,gd=ids.cuda(),donors.cuda();h=cache['h'][gi];m=cache['m'][gi];md=cache['m'][gd];hybrid=h-m+md;values={k:[] for k in ('native','graph','separate')};term_values={k:[] for k in ('graph','separate')}
   for z,hh,mm in ((cache['z'][gi],h,m),(cache['z'][gd],hybrid,md)):
    z,hh,mm=z.double(),hh.double(),mm.double();scale=(hh.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[:,None];native=((hh@A-.5*(mm@A))/scale-alpha)*((mm@B)/scale-beta);values['native'].append(native[:,2]);values['graph'].append(component_scalars(z,hh,graph)[:,2]);values['separate'].append((residual_write(z,hh,baseline['2'])@writer)/writer.square().sum())
    true_reads=torch.stack((mm@A[:,2],mm@B[:,2]),-1)
    for candidate,reads in (('graph',source_reads(z,graph)[...,4:6]),('separate',baseline_reads(z,baseline['2']))):
     terms=split(hh@A[:,2],scale[:,0],alpha[2],beta[2],true_reads,reads);term_values[candidate].append(terms);actual=values[candidate][-1]-native[:,2];term_checks.append(float((terms.sum(-1)-actual).norm()/actual.norm().clamp_min(1e-30)))
   for family,index in (('natural',0),('hybrid',1),('change',None)):
    vv={k:(v[index] if index is not None else v[1]-v[0]) for k,v in values.items()};reference=vv['native']
    for doc in (ids//240).unique().tolist():
     for cohort in ('all','continuation','spaced_word'):
      mask=ids//240==doc
      if cohort!='all':mask &=torch.tensor([labels[i][cohort] for i in ids.tolist()])
      mask=mask.cuda();count=int(mask.sum())
      if not count:continue
      for candidate in ('graph','separate'):
       terms=(term_values[candidate][index] if index is not None else term_values[candidate][1]-term_values[candidate][0])[mask];gram=terms.T@terms
       records.append(dict(gram=gram.tolist(),term_sums=terms.sum(0).tolist(),panel=version,document=documents[doc],domain='stdlib',selection='mode3',family=family,cohort=cohort,candidate=candidate,sites=count,error_energy=float((vv[candidate][mask]-reference[mask]).square().sum()),reference_energy=float(reference[mask].square().sum())))
 summaries=[]
 prior=json.loads((P/'FRONTIER_SCALAR_DIAGNOSTIC_V1.json').read_text())
 energy_replay=[]
 for family in ('natural','hybrid','change'):
  for cohort in ('all','continuation','spaced_word'):
   result=dict(family=family,cohort=cohort)
   for candidate in ('graph','separate'):
    rows=[r for r in records if r['family']==family and r['cohort']==cohort and r['candidate']==candidate]
    gram=sum((torch.tensor(r['gram'],dtype=torch.float64) for r in rows),torch.zeros(3,3,dtype=torch.float64))
    energy=sum(r['error_energy'] for r in rows)
    old=sum(r['error_energy'] for r in prior['records'] if r['family']==family and r['cohort']==cohort and r['candidate']==candidate)
    energy_replay.extend([abs(float(gram.sum())-energy)/max(energy,1e-30),abs(energy-old)/max(old,1e-30)])
    result[candidate]=dict(gram=gram.tolist(),total_energy=energy,diagonal_energy=float(gram.diag().sum()),cross_energy=float(gram.sum()-gram.diag().sum()))
   summaries.append(result)
 failures=[r for r in summaries if r['cohort']=='spaced_word' and r['family'] in ('natural','hybrid')]
 pred=dict(pred_a_replay=max(term_checks+energy_replay)<1e-8,pred_b_second_read=all(r['graph']['gram'][1][1]>r['separate']['gram'][1][1] for r in failures),pred_c_frozen=True)
 out=dict(plan=plan,records=records,summaries=summaries,source_replay=max(checks),term_replay=max(term_checks),energy_replay=max(energy_replay),predictions=pred,seconds=time.monotonic()-start,term_order=['first_read','second_read','error_product'])
 (P/'FRONTIER_READ_ERROR_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(predictions=pred,summaries=summaries)),flush=True)
if __name__=='__main__':main()
