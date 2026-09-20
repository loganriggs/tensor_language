#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fresh pred_b_context pred_c_capture
"""Frozen candidate validation on unused documents and longer context.
pred_a_fresh: fresh64 quartic error<.22, quadratic<.26,quartic better.
pred_b_context: context256 neither error exceeds its fresh64 error by>.05.
pred_c_capture: exact16forwards,16384? actual rows10240, disjoint token rows,
finite outputs and FP32/64 folded teacher disagreement<1e-3.
Price quadratic34560/4products,quartic47312/26products. No fitting or selection.
Null: compact programs fail new documents or longer contexts. Context shift is
not broad domain OOD. Full native model generates inputs; only folded path scored.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
PLAN=dict(cache='fineweb_n192_skip7000.pt',document_start=32,documents=32,contexts=[64,256],batch=4,forwards=16,rows=10240)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 import circuit_fast_screen_producer as producer
 from disk_guard import guard_torch_save
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=P/'FROZEN_FRESH_VALIDATION_V1.json';assert not out.exists();start=time.perf_counter();model=producer.Bilin18TorchBackend.load('cuda').model
 qs=torch.load(P/'NATIVE_GAUSSIAN_LINEAR_CONTROL_V1.pt',weights_only=True);hs=torch.load(P/'NATIVE_QUARTIC_MEAN_V1.pt',weights_only=True);q={k:v.cuda().double() for k,v in qs['programs']['centered'].items()};h={k:v.cuda().double() for k,v in hs['programs'][8].items()};state=model.state_dict();_,ru=torch.linalg.qr(state['lm_head.weight'].float());teacher=[ru@state['transformer.h.17.mlp.Down.weight'].float()/qs['teacher_scale'],state['transformer.h.17.mlp.Left.weight'].float(),state['transformer.h.17.mlp.Right.weight'].float(),state['transformer.h.16.mlp.Down.weight'].float()*state['transformer.h.17.lambdas'][0].float(),state['transformer.h.16.mlp.Left.weight'].float(),state['transformer.h.16.mlp.Right.weight'].float()]
 cache=torch.load(BQ/'.rowcache'/PLAN['cache'],weights_only=True);ids=cache[32:64];old={bytes(t.numpy().tobytes()) for t in cache[:32,:65]};assert all(bytes(t.numpy().tobytes()) not in old for t in ids[:,:65]);records=[];saved=[];calls=0
 def target(x,dtype):
  C,l,r,D,A,B=[t.to(dtype) for t in teacher];x=x.to(dtype);z=((x@A.T)*(x@B.T))@D.T;return ((z@l.T)*(z@r.T))@C.T
 for length in PLAN['contexts']:
  vals=[]
  def capture(_m,args):vals.append(args[0].detach().reshape(-1,1152).clone())
  handle=model.transformer.h[16].mlp.register_forward_pre_hook(capture)
  try:
   for row in range(0,32,4):
    tokens=ids[row:row+4,:length+1].cuda();model(tokens[:,:-1],tokens[:,1:].contiguous());calls+=1
  finally:handle.remove()
  x=torch.cat(vals).double();ys=[];ps=[];hscores=[]
  for z in x.split(256):
   ys.append(target(z,torch.float64));delta=z-q['mu'];ps.append(q['constant']+(delta@q['linear_reader'].T)@q['linear_writer'].T+((delta@q['quadratic_left'].T)*(delta@q['quadratic_right'].T))@q['quadratic_writer'].T);bank=((z@h['U'].flatten(0,1).T)*(z@h['V'].flatten(0,1).T)).reshape(len(z),4,4).sum(2);i,j=torch.triu_indices(4,4,device='cuda');hscores.append(((bank[:,i]*bank[:,j])@h['Z'].T)@h['W'].T+h['constant'])
  y=torch.cat(ys);preds=dict(quadratic=torch.cat(ps),quartic=torch.cat(hscores));metrics={name:dict(error=float((v-y).norm()/y.norm()),variation=float(((v-v.mean(0))-(y-y.mean(0))).norm()/(y-y.mean(0)).norm())) for name,v in preds.items()};precision=float((target(x[:32],torch.float32).double()-y[:32]).norm()/y[:32].norm());record=dict(context=length,rows=len(x),metrics=metrics,precision=precision,finite=bool(torch.isfinite(y).all()) and all(bool(torch.isfinite(v).all()) for v in preds.values()),token_sha256=hashlib.sha256(ids[:,:length+1].numpy().tobytes()).hexdigest());records.append(record);saved.append(dict(context=length,rows=x.float().cpu(),targets=y.float().cpu()));print(json.dumps(record),flush=True)
 a,b=records;pred=dict(pred_a_fresh=a['metrics']['quartic']['error']<.22 and a['metrics']['quadratic']['error']<.26 and a['metrics']['quartic']['error']<a['metrics']['quadratic']['error'],pred_b_context=all(b['metrics'][k]['error']<=a['metrics'][k]['error']+.05 for k in ['quadratic','quartic']),pred_c_capture=calls==16 and sum(r['rows'] for r in records)==10240 and all(r['finite'] and r['precision']<1e-3 for r in records));guard_torch_save(dict(panels=saved,teacher_scale=qs['teacher_scale']),str(P/'FROZEN_FRESH_VALIDATION_V1.pt'));out.write_text(json.dumps(dict(plan=PLAN,records=records,predictions=pred,calls=calls,seconds=time.perf_counter()-start,scope='Frozen programs, disjoint documents from previous32,64 vs256 context. Same source domain, no broad OOD claim; isolated folded contribution, not full model substitution.'),indent=2)+'\n');print(pred,flush=True)
if __name__=='__main__':main()
