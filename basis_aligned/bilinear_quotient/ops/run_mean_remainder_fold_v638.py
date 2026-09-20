#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_native_fold pred_b_cross_sufficient pred_c_normalization_small
"""Fold head1.8 mean/remainder jointly through MLP1, all four native corners.
Same now-opened v637 prompts.64forwards/0fits. Cross numerator plus exact RMS
correction; normalized bilinear face identity independently CPU-tested.
"""
import os,sys,json,time
from pathlib import Path
from datetime import datetime,timezone
from run_mean_head_effect_v637 import prompts
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/mean_remainder_fold_v638_result.json'
PREDICTIONS=dict(pred_a_native_fold='native write and interaction replay<=1e-4',
 pred_b_cross_sufficient='cross alone interaction error<=.20 every family',
 pred_c_normalization_small='normalization correction norm<=.20 interaction norm every family')


def main():
 plan=dict(forwards_max=64,fit_parameters=0,model_updates=0,model_backwards=0,execution_policy='managed_queue_only',
  families=['code','arithmetic','repetition','opened_prose'],modes=['native','mean','remainder','zero'])
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch,tiktoken
 import circuit_fast_screen_producer as producer
 import disk_guard
 sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
 from running_mean_head import execute
 from normalized_bilinear_face import decompose
 torch.set_grad_enabled(False);torch.set_num_threads(8);tic=time.perf_counter()
 model=producer.Bilin18TorchBackend.load('cuda').model;block=model.transformer.h[1];attn=block.attn
 original=attn.squared_attention;writer=attn.c_proj.weight[:,1024:1152].float()
 state=dict(mode='native');cache={};hooks=[]
 def patched(q,k,v,q2,k2):
  out=original(q,k,v,q2,k2);m=execute(v[:,:,8]);e=out[:,8]-m
  if state['mode']=='native':cache.update(mean=m@writer.T,remainder=e@writer.T);return out
  result=out.clone();result[:,8]={'mean':m,'remainder':e,'zero':torch.zeros_like(m)}[state['mode']];return result
 def start(_mod,args):cache['base']=block.lambdas[0]*args[0]+block.lambdas[1]*args[2]
 def after_attn(_mod,args,out):cache['h']=cache['base']+out[0]
 def after_mlp(_mod,args,out):cache['write']=out.detach()
 hooks.extend([block.register_forward_pre_hook(start),attn.register_forward_hook(after_attn),block.mlp.register_forward_hook(after_mlp)])
 def block_hook(index):
  def capture(_mod,args,out):cache['layers'][index]=out[0].detach()
  return capture
 hooks.extend(b.register_forward_hook(block_hook(i)) for i,b in enumerate(model.transformer.h))
 attn.squared_attention=patched
 L,R,D=[x.double() for x in [block.mlp.Left.weight,block.mlp.Right.weight,block.mlp.Down.weight]]
 bias=block.mlp.Down_bias.double();eps=torch.finfo(torch.float32).eps
 enc=tiktoken.get_encoding('gpt2');samples=[]
 for family,texts in prompts().items():
  samples.extend((family,torch.tensor(enc.encode(t),device='cuda')[None]) for t in texts)
 prose=torch.load(BQ/'.rowcache/fineweb_n192_skip11000.pt',weights_only=True)[32:36,:129].cuda()
 samples.extend(('opened_prose',row[None]) for row in prose)
 stats={};write_checks=[];interaction_checks=[];forwards=0
 try:
  for family,ids in samples:
   corners={};sources=None
   for mode in plan['modes']:
    state['mode']=mode;cache['layers']={}
    loss=float(model(ids[:,:-1],ids[:,1:].contiguous()));forwards+=1
    corners[mode]=dict(write=cache['write'].double(),layers=dict(cache['layers']),loss=loss)
    if mode=='native':sources=[cache[k].double() for k in ['h','mean','remainder']]
   h,m,e=sources;b=h-m-e
   terms=decompose(L,R,D,b,m,e,eps)
   modes=['zero','mean','remainder','native']
   for mode,write in zip(modes,terms['writes']):
    target=corners[mode]['write'];write_checks.append(float((write+bias-target).norm()/target.norm()))
   actual=corners['native']['write']-corners['mean']['write']-corners['remainder']['write']+corners['zero']['write']
   interaction_checks.append(float((terms['cross']+terms['normalization']-actual).norm()/actual.norm()))
   a=stats.setdefault(family,dict(target=0.,cross=0.,normalization=0.,cross_error=0.,cross_dot=0.,normalization_dot=0.,layers=[dict(interaction=0.,native=0.) for _ in range(18)]))
   for key,tensor in [('target',actual),('cross',terms['cross']),('normalization',terms['normalization']),('cross_error',terms['cross']-actual)]:a[key]+=float(tensor.square().sum())
   a['cross_dot']+=float((terms['cross']*actual).sum());a['normalization_dot']+=float((terms['normalization']*actual).sum())
   for layer in range(18):
    y={mode:corners[mode]['layers'][layer].double() for mode in modes}
    diff=y['native']-y['mean']-y['remainder']+y['zero']
    a['layers'][layer]['interaction']+=float(diff.square().sum());a['layers'][layer]['native']+=float(y['native'].square().sum())
   print(f'{family} write replay {max(write_checks):.3g} interaction replay {max(interaction_checks):.3g}',flush=True)
 finally:
  attn.squared_attention=original
  for hook in hooks:hook.remove()
 for a in stats.values():
  a.update(cross_relative_error=(a['cross_error']/a['target'])**.5,cross_norm_ratio=(a['cross']/a['target'])**.5,
   normalization_norm_ratio=(a['normalization']/a['target'])**.5,cross_aligned_fraction=a['cross_dot']/a['target'],normalization_aligned_fraction=a['normalization_dot']/a['target'])
  for row in a['layers']:row['relative_interaction_norm']=(row['interaction']/row['native'])**.5
 predictions=dict(pred_a_native_fold=max(write_checks+interaction_checks)<=1e-4,
  pred_b_cross_sufficient=all(a['cross_relative_error']<=.2 for a in stats.values()),
  pred_c_normalization_small=all(a['normalization_norm_ratio']<=.2 for a in stats.values()))
 assert forwards<=plan['forwards_max']
 result=dict(plan=plan,stats=stats,predictions=predictions,maximum_write_replay=max(write_checks),maximum_interaction_replay=max(interaction_checks),
  forwards=forwards,seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat(),
  scope='fold and response on opened prompts; local native MLP1 identity, not full-suffix causal explanation or simple circuit')
 disk_guard.guard_write(1000000,label='v638');OUT.write_text(json.dumps(result,indent=2)+'\n');print(predictions)


if __name__=='__main__':main()
