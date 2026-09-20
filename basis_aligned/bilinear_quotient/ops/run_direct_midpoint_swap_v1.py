#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_features pred_c_joint
"""Cross-document interchange of extracted midpoint16product features.
pred_a_instrument hashes/donoridentity/self-edit exact and same-token coverage>=.2;
pred_b_features same-token centered effect error<.3,cos>.95 all4scalars bothdomains;
pred_c_joint same-token joint centered error<.2 bothdomains.
Null removal fidelity fails contextual differences. Price48nativeforwards,
2donorfamilies*5edits*2amplitudes per recipient document, nofitting.
Native scalar amplitudes include their own source RMS; swap these amplitudes directly,
no recipient denominator reapplied. Mean cancels. Background and writers fixed.
"""
import os,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PANEL_PREFIX='SELECTIVE_CONFIRMATION'
OUTPUT_NAME='MIDPOINT_SWAP_V1.json'
DONOR_PREFIX='SELECTIVE_CONFIRMATION_DONORS'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=48,features=4,products=16,donor_families=['aligned','same_token'])));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from logit_effect_partition import partition
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter();out=P/OUTPUT_NAME;assert not out.exists()
 model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');e=torch.load(P/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);e={k:v.cuda().double() for k,v in e.items()};writer=torch.linalg.solve(ru,e['reduced_writers']);L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();C=e['scalar_readers'].T@ru@b17.mlp.Down.weight.double();records=[];checks=[];coverage=[];donors=torch.load(P/f'{DONOR_PREFIX}_V1.pt',weights_only=True);meta=json.loads((P/f'{DONOR_PREFIX}_V1.json').read_text())
 logits=lambda x:30*torch.tanh(model.lm_head(F.rms_norm(x,(1152,)))/30)
 for domain in ['fineweb','code']:
  tokens=torch.load(P/f'{PANEL_PREFIX}_{domain.upper()}_V1.pt',weights_only=True);manifest=next(r for r in meta['records'] if r['domain']==domain);assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==manifest['token_sha256'];coverage.append(manifest['same_token_coverage']);states=[];aa=[];bb=[]
  for row in tokens:
   c=capture(model,row[None,:256].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);s=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-m/2)/s;m=m/s;aa.append(((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@C.T-e['offset']);bb.append(((n@e['A'])*(m@e['B']))@e['readout']-e['offset']);states.append(c['final'].flatten(0,1))
  states=torch.cat(states);a=torch.cat(aa);b=torch.cat(bb);flat=tokens[:,:256].reshape(-1);doc=torch.arange(len(tokens)).repeat_interleave(256);targets=tokens[:,1:257].reshape(-1).cuda();checks.append(float(((a-a)@writer.T).abs().max()))
  for family in ['aligned','same_token']:
   mapping=donors[domain][family];valid=mapping>=0;assert torch.all(doc[valid]!=doc[mapping[valid]])
   if family=='same_token':assert torch.all(flat[valid]==flat[mapping[valid]])
   for document in range(len(tokens)):
    ids=torch.where(valid&(doc==document))[0].cuda();dst=mapping[ids.cpu()].cuda()
    if not len(ids):continue
    state=states[ids];native=logits(state);basece=F.cross_entropy(native,targets[ids],reduction='none')
    for mode in [0,1,2,3,'joint']:
     sl=list(range(4)) if mode=='joint' else [mode];effects=[];r=dict(domain=domain,family=family,document=document,mode=str(mode),sites=len(ids))
     for label,amp in [('native',a),('predicted',b)]:
      delta=((amp[dst][:,sl]-amp[ids][:,sl])@writer[:,sl].T).float();z=logits(state+delta);effects.append((z-native).double());r[label+'_ce_added']=float((F.cross_entropy(z,targets[ids],reduction='none')-basece).mean())
     r.update(partition(*effects));records.append(r)
 summary={}
 for domain in ['fineweb','code']:
  summary[domain]={}
  for family in ['aligned','same_token']:
   summary[domain][family]={}
   for mode in ['0','1','2','3','joint']:
    rr=[r for r in records if r['domain']==domain and r['family']==family and r['mode']==mode];sites=sum(r['sites'] for r in rr);en=sum(r['native_centered_effect_energy'] for r in rr);ep=sum(r['predicted_centered_effect_energy'] for r in rr);dot=sum(r['centered_effect_dot'] for r in rr);err=sum(r['centered_effect_error_energy'] for r in rr);summary[domain][family][mode]=dict(centered_effect_relative_error=(err/en)**.5,centered_effect_cosine=dot/(en*ep)**.5,sites=sites,native_centered_effect_energy=en,centered_effect_error_energy=err,native_ce_added=sum(r['native_ce_added']*r['sites'] for r in rr)/sites,predicted_ce_added=sum(r['predicted_ce_added']*r['sites'] for r in rr)/sites)
 pred=dict(pred_a_instrument=max(checks)==0 and min(coverage)>=.2,pred_b_features=all(summary[d]['same_token'][m]['centered_effect_relative_error']<.3 and summary[d]['same_token'][m]['centered_effect_cosine']>.95 for d in summary for m in ['0','1','2','3']),pred_c_joint=all(summary[d]['same_token']['joint']['centered_effect_relative_error']<.2 for d in summary))
 result=dict(predictions=pred,summary=summary,records=records,seconds=time.perf_counter()-start,scope='Already-normalized operational feature-amplitude interchange. Explicit fixed writer and recipient native background. Reused panels/donor maps; no semantic/task-selectivity claim.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
