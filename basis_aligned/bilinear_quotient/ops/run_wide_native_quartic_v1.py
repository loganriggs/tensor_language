#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_capacity pred_c_target
"""Widths4/32, inherited/random,100Adamsteps. True native purequartic targets.
Integrity cachedtarget/FP32export<1e-4; bestwide second-panel error<=.8bestnarrow;
onewide fit/evalboth<5%. Same2048cachedrows perpanel, not newOOD. Weight-frame
replay and all coefficients priced; no residual/bias/normalization replacement.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,widths=[4,32],starts=['inherited','random'],steps=100)));return
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;sys.path.insert(0,str(P))
 from empirical_quartic_dictionary import fit,evaluate
 start=time.monotonic();out=P/'WIDE_NATIVE_QUARTIC_V1.json';assert not out.exists();control=json.loads((P/'EMPIRICAL_QUARTIC_DICTIONARY_CONTROLS_V1.json').read_text());long=json.loads((P/'EMPIRICAL_QUARTIC_DICTIONARY_LONG_V1.json').read_text());assert all(r['validation_error']<.01 for r in long['warm_start_controls']);assert max(max(r['solve_error'],r['gradient_error']) for r in control['controls'])<1e-9
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];xs=[r['rows'].cuda().double() for r in panels];ys=[t.cuda().double() for t in data['targets']];scale=data['teacher_scale'];old=torch.load(P/'NATIVE_MIXED_ROOT_V2.pt',weights_only=True)['students'][('adam',.005,1)]
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight')/scale;L,R=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A,B=w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight');x=xs[0][:32].float();h=((x@A.T)*(x@B.T))@D.T;reference=((h@L.T)*(h@R.T))@C.T;target_replay=float((reference.double()-ys[0][:32]).norm()/ys[0][:32].norm());assert target_replay<1e-4
 del C,L,R,D,A,B,h,reference,x;records=[]
 for width in [4,32]:
  for init in ['inherited','random']:
   torch.manual_seed(2719);u=torch.randn(width,4,1152,device='cuda',dtype=torch.float64)/math.sqrt(1152);v=torch.randn_like(u)/math.sqrt(1152)
   if init=='inherited':u[:4]=old['U'].cuda().double();v[:4]=old['V'].cuda().double()
   rate=.03*math.sqrt(4/1152)
   row,program=fit(xs[0],ys[0],width,steps=100,rate=rate,optimizer=control['selected_optimizer'],initial=(u,v));evaluation=float((evaluate(program,xs[1])-ys[1]).norm()/ys[1].norm())
   with torch.no_grad():
    archive={k:t.cpu().float() for k,t in program.items()};archive['C']=torch.linalg.solve(ru.double(),program['C']*scale).cpu().float();physical={k:t.cuda() for k,t in archive.items()};original=evaluate(program,xs[0][:128]);actual=evaluate(physical,xs[0][:128].float()).double()@ru.double().T/scale;drift=float((actual-original).norm()/original.norm());path=P/f'WIDE_NATIVE_QUARTIC_{width}_{init.upper()}_V1.pt';torch.save(archive,path)
   row.update(width=width,learning_rate=rate,initialization=init,evaluation_error=evaluation,export_replay=drift,products=4*width+width*(width+1)//2,stored_coefficients=sum(t.numel() for t in archive.values()),program_sha256=hashlib.sha256(path.read_bytes()).hexdigest());records.append(row);print(json.dumps(row),flush=True)
 wide=[r for r in records if r['width']==32];narrow=[r for r in records if r['width']==4]
 pred=dict(pred_a_integrity=target_replay<1e-4 and all(math.isfinite(r['objective']) and r['export_replay']<1e-4 for r in records),pred_b_capacity=min(r['evaluation_error'] for r in wide)<=.8*min(r['evaluation_error'] for r in narrow),pred_c_target=any(r['training_error']<.05 and r['evaluation_error']<.05 for r in wide))
 result=dict(predictions=pred,records=records,target_replay=target_replay,optimizer=control['selected_optimizer'],seconds=time.monotonic()-start,scope='True native purequartic teacher, empirical functional fit on old2048calibrationstates; oldsecondpanel reporting only. Best trainingobjective checkpoint; physical residual writer exported with teacher_scale/QR folded. Counts exclude common unembedding, no explicit-normalization/residual-path replacement or circuit adoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(predictions=pred,seconds=result['seconds'])),flush=True)
if __name__=='__main__':main()
