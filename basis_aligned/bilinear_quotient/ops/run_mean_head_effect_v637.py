#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_effect_prediction pred_c_effect_direction pred_d_loss_prediction
"""Frozen parameter-free mean predicts head1.8 removal, including shifted prompts.

Compare actual head-zero removal with subtracting the extracted mean from the
native head, through the entire actual suffix. No fitted scalars or ranking.
Fresh constructed prompts relative to this candidate, not pretraining-OOD.
64 forwards, zero updates/fits. Low replacement CE alone cannot pass effect gate.
"""
import os,sys,time,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/mean_head_effect_v637_result.json'
PREDICTIONS=dict(pred_a_instrument='native intervention identity error<=2e-5',
 pred_b_effect_prediction='centered logit removal relative error<=.20 all families',
 pred_c_effect_direction='removal cosine>=.90 all families',
 pred_d_loss_prediction='mean absolute per-token NLL effect error<=.02 all families')


def prompts():
 return dict(code=[f'def scale_{i}(items):\n    total = 0\n    for value in items:\n        total += value * {i}\n    return total\n\nresult = scale_{i}([2, 5, 8])\n' for i in [11,13,17,19]],
 arithmetic=[f'The ledger starts with {i} coins. We add {i+7}, remove 3, and add 12. Now the ledger has {2*i+16} coins. A second ledger starts with {i+4} coins.' for i in [23,31,47,59]],
 repetition=[' '.join([w]*45)+'. A different word follows.' for w in ['copper','violet','triangle','repeat']])


def main():
 plan=dict(texts=prompts(),prose_rows=4,forwards_max=64,model_updates=0,model_backwards=0,fit_parameters=0,
  execution_policy='managed_queue_only',modes=['native','zero_head','subtract_program','program_only'])
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch,tiktoken
 import torch.nn.functional as F
 import circuit_fast_screen_producer as producer
 import disk_guard
 sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
 from running_mean_head import execute
 torch.set_grad_enabled(False);torch.set_num_threads(8);tic=time.perf_counter()
 model=producer.Bilin18TorchBackend.load('cuda').model;attn=model.transformer.h[1].attn;original=attn.squared_attention
 state=dict(mode='native',logits=None,closure=[])
 def patched(q,k,v,q2,k2):
  out=original(q,k,v,q2,k2);mode=state['mode'];pred=execute(v[:,:,8])
  if mode=='native':
   state['closure'].append(float(((out[:,8]-pred)+pred-out[:,8]).norm()/out[:,8].norm()));return out
  result=out.clone()
  if mode=='zero_head':result[:,8]=0
  elif mode=='subtract_program':result[:,8]=out[:,8]-pred
  else:result[:,8]=pred
  return result
 def capture(_m,_args,out):state['logits']=30*torch.tanh(out.float()/30)
 attn.squared_attention=patched;hook=model.lm_head.register_forward_hook(capture)
 enc=tiktoken.get_encoding('gpt2');samples=[]
 for family,texts in plan['texts'].items():
  for text in texts:samples.append((family,torch.tensor(enc.encode(text),device='cuda')[None]))
 prose=torch.load(BQ/'.rowcache/fineweb_n192_skip11000.pt',weights_only=True)[32:36,:129].cuda()
 samples.extend(('opened_prose',row[None]) for row in prose)
 rows=[];aggregate={};forwards=0
 try:
  for family,ids in samples:
   tokens,target=ids[:,:-1],ids[:,1:].contiguous();logits={};losses={};nll={}
   for mode in plan['modes']:
    state['mode']=mode;losses[mode]=float(model(tokens,target));forwards+=1
    logits[mode]=state['logits'];nll[mode]=F.cross_entropy(logits[mode].flatten(0,1),target.flatten(),reduction='none')
   actual=logits['zero_head']-logits['native'];predicted=logits['subtract_program']-logits['native']
   actual=actual-actual.mean(-1,keepdim=True);predicted=predicted-predicted.mean(-1,keepdim=True)
   a=aggregate.setdefault(family,dict(error=0.,target=0.,pred=0.,dot=0.,nll_abs=0.,tokens=0,sign_good=0,sign_total=0))
   a['error']+=float((actual-predicted).double().square().sum());a['target']+=float(actual.double().square().sum())
   a['pred']+=float(predicted.double().square().sum());a['dot']+=float((predicted.double()*actual.double()).sum())
   effect=nll['zero_head']-nll['native'];prediction=nll['subtract_program']-nll['native'];live=effect.abs()>.01
   a['nll_abs']+=float((effect-prediction).abs().sum());a['tokens']+=tokens.numel()
   a['sign_good']+=int(((effect.sign()==prediction.sign())&live).sum());a['sign_total']+=int(live.sum())
   row=dict(family=family,tokens=tokens.numel(),token_sha256=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
    ce=losses,zero_head_ce_added=losses['zero_head']-losses['native'],program_only_ce_added=losses['program_only']-losses['native'])
   rows.append(row);print(json.dumps(row),flush=True)
 finally:attn.squared_attention=original;hook.remove()
 metrics={f:dict(**a,relative_logit_effect_error=(a['error']/a['target'])**.5,
  effect_cosine=a['dot']/(a['target']*a['pred'])**.5,nll_effect_mae=a['nll_abs']/a['tokens'],
  nll_sign_agreement=a['sign_good']/a['sign_total'] if a['sign_total'] else None) for f,a in aggregate.items()}
 predictions=dict(pred_a_instrument=max(state['closure'])<=2e-5,
  pred_b_effect_prediction=all(a['relative_logit_effect_error']<=.2 for a in metrics.values()),
  pred_c_effect_direction=all(a['effect_cosine']>=.9 for a in metrics.values()),
  pred_d_loss_prediction=all(a['nll_effect_mae']<=.02 for a in metrics.values()))
 assert forwards<=plan['forwards_max']
 result=dict(plan=plan,rows=rows,metrics=metrics,predictions=predictions,forwards=forwards,maximum_algebraic_closure=max(state['closure']),
  seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat(),
  scope='full native suffix interventions; no semantic selectivity claim; novel constructed prompts relative to candidate, opened prose; no pretraining-OOD claim')
 disk_guard.guard_write(1000000,label='v637');OUT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(metrics,indent=2));print(predictions)


if __name__=='__main__':main()
