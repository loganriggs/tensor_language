#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_scalar_failures pred_c_frozen
"""Revisit opened32code files: scalar versus logit-effect weighting, no fitting.
Pred_a native scalar/source replay<1e-4. Pred_b both pooled logit-failing
mode3/spaced-word cells also exceed1.1baseline in unweighted scalar error.
Pred_c candidate/baseline and panel hashes unchanged. Counterhypothesis:
final normalization/logit sensitivity reweights a passing scalar approximation.
32context256captures; use original cross-document same-token donor maps.
"""
import os,sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 sys.path.insert(0,str(P));from pairwise_component_interface import component_scalars,move_preserving_dtype
 from source_interface import residual_write
 plan=json.loads((P/'FRONTIER_SCALAR_DIAGNOSTIC_PLAN_V1.json').read_text())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  torch.set_num_threads(2);graph=torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True);z=torch.ones(2,3,1152,dtype=torch.float64);assert component_scalars(z,z,graph).shape==(2,3,3);print('32captures, same frozen graph; batched scalar preflight PASS');return
 for name,digest in plan['hashes'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
 import torch.nn.functional as F
 import tiktoken
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from token_boundary_conditions import annotate
 torch.set_num_threads(2);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();model=Bilin18TorchBackend.load('cuda').model.float();graph=move_preserving_dtype(torch.load(P/'FRONTIER_FRESH_GRAPH_V1.pt',weights_only=True),'cuda');baseline=move_preserving_dtype(torch.load(P/'FRONTIER_FRESH_BASELINE_CALIBRATION_SHAPED_V1.pt',weights_only=True),'cuda');fold=move_preserving_dtype(torch.load(P/'PARTIAL_GRAPH_ORIGINAL_FORMS_V1.pt',weights_only=True),'cuda');A,B=fold['A'],fold['B'];alpha,beta=fold['alpha'],fold['beta'];writer=graph['residual_writer'];checks=[];records=[];enc=tiktoken.get_encoding('gpt2')
 for version in (1,2):
  tokens=torch.load(P/f'FRONTIER_FRESH_TOKENS_V{version}.pt',weights_only=True)['stdlib'];mapping=torch.load(P/f'FRONTIER_FRESH_DONORS_V{version}.pt',weights_only=True)['stdlib'];documents=json.loads((P/f'FRONTIER_FRESH_PLAN_V{version}.json').read_text())['recipient_documents']['stdlib'];cache={k:[] for k in ('z','h','m')};labels=[]
  for row in tokens:
   raw=capture(model,row[None,:256].cuda());z=raw['x16'].flatten(0,1)[16:];h=raw['h17'].flatten(0,1)[16:];m=(model.transformer.h[17].lambdas[0]*(raw['m16']-model.transformer.h[16].mlp.Down_bias)).flatten(0,1)[16:]
   q=torch.einsum('ni,oij,nj->no',z.double(),fold['matrices'],z.double());true=torch.stack([m.double()@A,m.double()@B],-1).flatten(1);checks.append(float((q-true).norm()/true.norm()))
   for key,value in (('z',z),('h',h),('m',m)):cache[key].append(value)
   labels+=annotate(row.tolist(),enc)[16:256]
  cache={k:torch.cat(v) for k,v in cache.items()};valid=(mapping>=0).nonzero().flatten();flat=tokens[:,16:256].flatten();assert torch.all(flat[valid]==flat[mapping[valid]]) and torch.all(valid//240!=mapping[valid]//240)
  for offset in range(0,len(valid),96):
   ids=valid[offset:offset+96];donors=mapping[ids];gi,gd=ids.cuda(),donors.cuda();h=cache['h'][gi];m=cache['m'][gi];md=cache['m'][gd];hybrid=h-m+md;values={k:[] for k in ('native','graph','separate')}
   for z,hh,mm in ((cache['z'][gi],h,m),(cache['z'][gd],hybrid,md)):
    z,hh,mm=z.double(),hh.double(),mm.double();scale=(hh.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt()[:,None];native=((hh@A-.5*(mm@A))/scale-alpha)*((mm@B)/scale-beta);values['native'].append(native[:,2]);values['graph'].append(component_scalars(z,hh,graph)[:,2]);values['separate'].append((residual_write(z,hh,baseline['2'])@writer)/writer.square().sum())
   for family,index in (('natural',0),('hybrid',1),('change',None)):
    vv={k:(v[index] if index is not None else v[1]-v[0]) for k,v in values.items()};reference=vv['native']
    for doc in (ids//240).unique().tolist():
     for cohort in ('all','continuation','spaced_word'):
      mask=ids//240==doc
      if cohort!='all':mask &=torch.tensor([labels[i][cohort] for i in ids.tolist()])
      mask=mask.cuda();count=int(mask.sum())
      if not count:continue
      for candidate in ('graph','separate'):records.append(dict(panel=version,document=documents[doc],domain='stdlib',selection='mode3',family=family,cohort=cohort,candidate=candidate,sites=count,error_energy=float((vv[candidate][mask]-reference[mask]).square().sum()),reference_energy=float(reference[mask].square().sum())))
 summaries=[]
 for family in ('natural','hybrid','change'):
  for cohort in ('all','continuation','spaced_word'):
   errors={k:sum(r['error_energy'] for r in records if r['family']==family and r['cohort']==cohort and r['candidate']==k) for k in ('graph','separate')};scalar_ratio=(errors['graph']/errors['separate'])**.5;prior=json.loads((P/'FRONTIER_FRESH_NATIVE_POOLED_V1.json').read_text())['summary']['stdlib']['mode3'];logit_ratio=prior['graph'][family][cohort]['effect_relative_error']/prior['separate'][family][cohort]['effect_relative_error'];summaries.append(dict(family=family,cohort=cohort,scalar_ratio=scalar_ratio,logit_ratio=logit_ratio,scalar_relative_pass=scalar_ratio<=1.1,logit_relative_pass=logit_ratio<=1.1))
 failures=[r for r in summaries if not r['logit_relative_pass']];pred=dict(pred_a_replay=max(checks)<1e-4,pred_b_scalar_failures=all(not r['scalar_relative_pass'] for r in failures),pred_c_frozen=True)
 out=dict(plan=plan,records=records,summaries=summaries,source_replay=max(checks),predictions=pred,seconds=time.monotonic()-start,scope='Opened-panel diagnostic, unchanged programs. Scalar ratios use uncentered true-scalar energy. Threshold flips describe endpoint weighting, not isolated causal attribution to RMSNorm versus softcap. No new candidate or fresh claim.');(P/'FRONTIER_SCALAR_DIAGNOSTIC_V1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(predictions=pred,summaries=summaries)),flush=True)
if __name__=='__main__':main()
