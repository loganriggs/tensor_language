#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_major pred_c_all
"""Frozen scalar interchange, FEATURE_SWAP_PLAN_V1.md.
pred_a_instrument: self-edit zero, donor identities valid, coverage>=.2.
pred_b_major: same-token modes0/1 cosine>.9/error<.4, effectRMS>1e-3, both domains.
pred_c_all: all4 same-token modes cosine>.8/error<.65, both domains.
Null: extracted removal agreement does not predict context-dependent interchange.
Price13916scalar+4608writer coefficients/10products, native background retained.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PROGRAM_FILE='EXTRACTED_SCALAR_INTERVENTIONS_V1.pt'
OUTPUT_FILE='FEATURE_SWAP_V1.json'
PANEL_PREFIX='BLEND_CONFIRMATION'
DONOR_PREFIX='FEATURE_SWAP_DONORS'
PLAN=dict(domains=['fineweb','code'],families=['aligned','same_token'],modes=[0,1,2,3,'joint'],context=256,native_forwards=48)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 import torch.nn.functional as F
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from native_quartic_branch import pure_branch
 from extract_scalar_modes import evaluate
 from logit_effect_partition import partition
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/OUTPUT_FILE;assert not out.exists();start=time.perf_counter();model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.float());ru=ru.double();artifact=torch.load(P/PROGRAM_FILE,weights_only=True);program={k:v.cuda().double() for k,v in artifact['program'].items()};writer=artifact['residual_writer'].cuda().double();U=torch.load(P/'CANONICAL_ROOT_FEATURES_V1.pt',weights_only=True)['output_directions'].cuda().double();scale=artifact['teacher_scale'];donors=torch.load(P/f'{DONOR_PREFIX}_V1.pt',weights_only=True);meta=json.loads((P/f'{DONOR_PREFIX}_V1.json').read_text());records=[];checks=[];coverage=[]
 logits=lambda state:30*torch.tanh(model.lm_head(F.rms_norm(state,(1152,)))/30)
 for domain in PLAN['domains']:
  tokens=torch.load(P/f'{PANEL_PREFIX}_{domain.upper()}_V1.pt',weights_only=True);manifest=next(r for r in meta['records'] if r['domain']==domain);assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==manifest['token_sha256'];coverage.append(manifest['same_token_coverage']);states=[];den=[];aa=[];bb=[]
  for row in tokens:
   cache=capture(model,row[None,:256].cuda());pure=pure_branch(cache['m16'],b16.mlp.Down_bias,b17.lambdas[0],b17.mlp.Left.weight,b17.mlp.Right.weight,b17.mlp.Down.weight).flatten(0,1).double();aa.append((pure@ru.T/scale)@U);bb.append(evaluate(program,cache['x16'].flatten(0,1).double()));states.append(cache['final'].flatten(0,1));den.append(cache['h17'].square().mean(-1,keepdim=True).flatten(0,1)+torch.finfo(cache['h17'].dtype).eps)
  states=torch.cat(states);den=torch.cat(den);a=torch.cat(aa);b=torch.cat(bb);flat=tokens[:,:256].reshape(-1);doc=torch.arange(len(tokens)).repeat_interleave(256);targets=tokens[:,1:257].reshape(-1).cuda();selfedit=(a-a)@writer.T;checks.append(float(selfedit.abs().max()))
  for family in PLAN['families']:
   mapping=donors[domain][family];valid=mapping>=0;assert torch.all(doc[valid]!=doc[mapping[valid]])
   if family=='same_token':assert torch.all(flat[valid]==flat[mapping[valid]])
   for document in range(len(tokens)):
    ids=torch.where(valid&(doc==document))[0].cuda();dst=mapping[ids.cpu()].cuda()
    if not len(ids):continue
    state=states[ids];native=logits(state);logp=F.log_softmax(native,dim=-1);prob=logp.exp();basece=F.cross_entropy(native,targets[ids],reduction='none')
    for g in PLAN['modes']:
     sl=list(range(4)) if g=='joint' else [g];effects=[];ces=[];r=dict(domain=domain,family=family,document=document,mode=g,sites=len(ids))
     for label,amp in [('native',a),('predicted',b)]:
      delta=((amp[dst][:,sl]-amp[ids][:,sl])@writer[:,sl].T).float()/den[ids];z=logits(state+delta);effect=(z-native).double();ce=(F.cross_entropy(z,targets[ids],reduction='none')-basece).double();effects.append(effect);ces.append(ce);r[label+'_effect_energy']=float(effect.square().sum());r[label+'_ce_effect_energy']=float(ce.square().sum());r[label+'_ce_added']=float(ce.mean());r[label+'_kl']=float((prob*(logp-F.log_softmax(z,dim=-1))).sum(-1).mean())
     r['effect_dot']=float((effects[0]*effects[1]).sum());r['effect_error_energy']=float((effects[0]-effects[1]).square().sum());r['ce_effect_dot']=float((ces[0]*ces[1]).sum());r['ce_effect_error_energy']=float((ces[0]-ces[1]).square().sum());r.update(partition(*effects));records.append(r)
   print(domain,family,'done',flush=True)
 summary={}
 for domain in PLAN['domains']:
  summary[domain]={}
  for family in PLAN['families']:
   summary[domain][family]={}
   for g in PLAN['modes']:
    rr=[r for r in records if r['domain']==domain and r['family']==family and r['mode']==g];n=sum(r['sites'] for r in rr);en=sum(r['native_effect_energy'] for r in rr);ep=sum(r['predicted_effect_energy'] for r in rr);dot=sum(r['effect_dot'] for r in rr);err=sum(r['effect_error_energy'] for r in rr);cen=sum(r['native_ce_effect_energy'] for r in rr)
    summary[domain][family][str(g)]=dict(sites=n,effect_cosine=dot/(en*ep)**.5,effect_relative_error=(err/en)**.5,native_effect_rms=(en/(n*50304))**.5,ce_effect_relative_error=(sum(r['ce_effect_error_energy'] for r in rr)/cen)**.5,native_ce_added=sum(r['native_ce_added']*r['sites'] for r in rr)/n,predicted_ce_added=sum(r['predicted_ce_added']*r['sites'] for r in rr)/n)
    ec=sum(r['native_centered_effect_energy'] for r in rr);pc=sum(r['predicted_centered_effect_energy'] for r in rr);summary[domain][family][str(g)].update(centered_effect_relative_error=(sum(r['centered_effect_error_energy'] for r in rr)/ec)**.5,centered_effect_cosine=sum(r['centered_effect_dot'] for r in rr)/(ec*pc)**.5,native_common_fraction=sum(r['native_common_effect_energy'] for r in rr)/en)
 major=[summary[d]['same_token'][str(g)] for d in PLAN['domains'] for g in [0,1]];allm=[summary[d]['same_token'][str(g)] for d in PLAN['domains'] for g in range(4)]
 pred=dict(pred_a_instrument=max(checks)==0 and min(coverage)>=.2,pred_b_major=all(r['effect_cosine']>.9 and r['effect_relative_error']<.4 and r['native_effect_rms']>1e-3 for r in major),pred_c_all=all(r['effect_cosine']>.8 and r['effect_relative_error']<.65 for r in allm))
 result=dict(plan=PLAN,summary=summary,records=records,predictions=pred,self_edit_max=max(checks),seconds=time.perf_counter()-start,scope='Frozen context-dependent component interchange in original recipient background. Same-token control removes static token-only explanation; not semantic/task selectivity or a whole upstream-state swap.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(summary=summary,predictions=pred),indent=2))
if __name__=='__main__':main()
