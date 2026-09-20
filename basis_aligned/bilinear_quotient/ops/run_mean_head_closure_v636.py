#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_mean_reference pred_b_closed_diag pred_c_output_prediction pred_d_signed_removal
"""Close head1.8's native diagonal oracle; signed mean and removal controls.
49 native forwards maximum, one calibration scalar, no model updates.
Opened panels only. Extraction boundary is the native mixed-value stream.
"""
import os,sys,json,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];BQ=ROOT/'basis_aligned/bilinear_quotient'
OUT=BQ/'circuits/followups/mean_head_closure_v636_result.json'
PREDICTIONS=dict(pred_a_mean_reference='native-diagonal mean CE added<=.015',
 pred_b_closed_diag='frozen scalar diagonal CE added<=.015',
 pred_c_output_prediction='frozen full head relative L2 error<=.10',
 pred_d_signed_removal='zero and sign flip cost >= .02 more than frozen program')


def main():
 plan=dict(panels=['fineweb_n192_skip7000.pt','fineweb_n192_skip11000.pt'],lengths=[128,512],documents=16,
  modes=['native','mean_native_diag','mean_zero_diag','mean_fit_diag','zero_head','flip_mean'],
  forwards_max=49,fit_parameters=1,model_updates=0,model_backwards=0,execution_policy='managed_queue_only')
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(plan));return
 import torch
 import circuit_fast_screen_producer as producer
 import disk_guard
 sys.path.insert(0,str(ROOT/'basis_aligned/polynomial_causal'))
 from running_mean_head import execute
 torch.set_grad_enabled(False);torch.set_num_threads(8);tic=time.perf_counter()
 model=producer.Bilin18TorchBackend.load('cuda').model
 attn=model.transformer.h[1].attn;original=attn.squared_attention
 state=dict(mode='capture',diags=[],g=None,error=0.,target=0.,reference_error=0.,calls=0)
 def patched(q,k,v,q2,k2):
  out=original(q,k,v,q2,k2)
  h=8;hd=q.shape[-1]
  diagonal=((q[:,:,h]*k[:,:,h]).sum(-1)/hd)*((q2[:,:,h]*k2[:,:,h]).sum(-1)/hd)
  mode=state['mode']
  if mode=='capture':state['diags'].append(diagonal.detach());return out
  state['calls']+=1
  if mode=='native':return out
  values=v[:,:,h]
  if mode=='mean_native_diag':pred=execute(values,diagonal)
  elif mode=='mean_zero_diag':pred=execute(values,0.)
  elif mode=='zero_head':pred=torch.zeros_like(values)
  else:pred=execute(values,state['g'],-1. if mode=='flip_mean' else 1.)
  truth=out[:,h]
  state['error']+=float((pred-truth).double().square().sum());state['target']+=float(truth.double().square().sum())
  result=out.clone();result[:,h]=pred;return result
 attn.squared_attention=patched;forwards=0
 try:
  fit=torch.load(BQ/'.rowcache/fineweb_n480_skip80.pt',weights_only=True)[:16,:513].cuda()
  model(fit[:,:-1],fit[:,1:].contiguous());forwards+=1
  diagonal=torch.cat(state['diags']);state['g']=float(diagonal.mean());state['diags']=[]
  calibration=dict(mean=state['g'],std=float(diagonal.std()),min=float(diagonal.min()),max=float(diagonal.max()),tokens=diagonal.numel())
  rows=[]
  for panel in plan['panels']:
   tokens=torch.load(BQ/'.rowcache'/panel,weights_only=True)[:16].cuda()
   for length in plan['lengths']:
    assert tokens.shape[1]>=length+1
    native=None
    for mode in plan['modes']:
     state.update(mode=mode,error=0.,target=0.,calls=0);loss=0.
     for start in range(0,16,8):
      batch=tokens[start:start+8,:length+1]
      loss+=float(model(batch[:,:-1],batch[:,1:].contiguous()))/2;forwards+=1
     assert state['calls']==2
     if mode=='native':native=loss
     row=dict(panel=panel,length=length,mode=mode,ce=loss,ce_added=loss-native,
       output_squared_error=state['error'],output_squared_norm=state['target'],
       output_relative_error=(state['error']/state['target'])**.5 if state['target'] else 0.)
     rows.append(row);print(json.dumps(row),flush=True)
 finally:attn.squared_attention=original
 group=lambda mode:[r for r in rows if r['mode']==mode]
 references=group('mean_native_diag');closed=group('mean_fit_diag')
 signed=all(z['ce_added']>=c['ce_added']+.02 and f['ce_added']>=c['ce_added']+.02 for z,f,c in zip(group('zero_head'),group('flip_mean'),closed))
 predictions=dict(pred_a_mean_reference=max(r['ce_added'] for r in references)<=.015,
  pred_b_closed_diag=max(r['ce_added'] for r in closed)<=.015,
  pred_c_output_prediction=max(r['output_relative_error'] for r in closed)<=.1,pred_d_signed_removal=signed)
 assert forwards<=plan['forwards_max']
 result=dict(plan=plan,calibration=calibration,rows=rows,predictions=predictions,forwards=forwards,
  executable=dict(module='running_mean_head.py',mass=1.,diagonal=state['g'],input_port='head1.8 mixed value stream [B,T,128]',
   trainable_scalar_count=1,prefix_state_values=128,scope='excludes upstream value projections and downstream output projection'),
  seconds=time.perf_counter()-tic,finished_utc=datetime.now(timezone.utc).isoformat(),
  scope='opened panels; scalar fitted on separate calibration rows; no semantic selectivity claim; no whole-model extraction')
 disk_guard.guard_write(1000000,label='v636 result');OUT.write_text(json.dumps(result,indent=2)+'\n');print(predictions)


if __name__=='__main__':main()
