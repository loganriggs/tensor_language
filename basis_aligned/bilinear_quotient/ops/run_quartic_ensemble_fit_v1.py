#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_transfer pred_c_values
"""Fixedstate fourdonor fit, same32features/300Adamsteps.
Heldpaircal<=.8halfcontrol andeval<=.9halfcontrol; evalvalue<=1.1halfcontrol.
Replay/export<1e-4;656products903168coeff; moreperstepcompute, no OOD claim.
"""
import os,sys,json,time,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
TRAIN=[997,313,1427,179];HELD=[613,1379]
def main():
 sys.path.insert(0,str(P));import torch
 from quartic_finite_response import fit,response_features,ensemble_design
 from empirical_quartic_dictionary import evaluate
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  assert not set(TRAIN)&set(HELD);torch.manual_seed(31);x=torch.randn(23,7,dtype=torch.float64);dd=torch.randn(4,23,7,dtype=x.dtype);y=torch.randn(23,4,dtype=x.dtype);rr=torch.randn(4,23,4,dtype=x.dtype);init=(torch.randn(3,2,7,dtype=x.dtype),torch.randn(3,2,7,dtype=x.dtype));info,p=fit(x,y,dd,rr,init,steps=2,design_builder=ensemble_design);assert p['C'].shape==(4,6) and math.isfinite(info['objective']);print(json.dumps(dict(callback_smoke='pass',train_shifts=TRAIN,held_shifts=HELD,steps=300)));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'QUARTIC_ENSEMBLE_FIT_V1.json';assert not out.exists()
 data=torch.load(P/'NATIVE_VARIATION_AUDIT_V1.pt',weights_only=True);scale=data['teacher_scale'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];xs=[r['rows'].cuda().double() for r in panels];shifts=TRAIN+HELD;ds=[torch.stack([.5*(x.roll(s,0)-x) for s in shifts]) for x in xs]
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));C=ru@w('transformer.h.17.mlp.Down.weight')/scale;L,R=w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight');D=w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0];A,B=w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')
  def teacher(x):
   x=x.float();h=((x@A.T)*(x@B.T))@D.T;return (((h@L.T)*(h@R.T))@C.T).double()
  ys=[teacher(x) for x in xs];rs=[torch.stack([teacher(x+d)-y for d in dd]) for x,dd,y in zip(xs,ds,ys)];replay=max(float((y-old.cuda().double()).norm()/old.cuda().double().norm()) for y,old in zip(ys,data['targets']))
 initial0=torch.load(P/'QUARTIC_JOINT_READOUT_LAMBDA_1_V1.pt',weights_only=True);initial=(initial0['U'].cuda().double(),initial0['V'].cuda().double())
 info,program=fit(xs[0],ys[0],ds[0][:4],rs[0][:4],initial,steps=300,rate=.03*math.sqrt(4/1152),design_builder=ensemble_design)
 with torch.no_grad():
  def assess(p):
   rows=[]
   for x,dd,y,rr in zip(xs,ds,ys,rs):
    errors=[float((response_features(x,d,p['U'],p['V'])@p['C'].T-r).norm()/r.norm()) for d,r in zip(dd,rr)];rows.append(dict(value_error=float((evaluate(p,x)-y).norm()/y.norm()),response_errors=dict(zip(map(str,shifts),errors))))
   return rows
  old={k:v.cuda().double() for k,v in torch.load(P/'QUARTIC_RESPONSE_FEATURES_32_WEIGHT_1_V1.pt',weights_only=True).items()};old['C']=ru.double()@old['C']/scale;results=dict(ensemble=assess(program),half=assess(old))
  archive={k:v.cpu().float() for k,v in program.items()};archive['C']=torch.linalg.solve(ru.double(),program['C']*scale).cpu().float();physical={k:v.cuda() for k,v in archive.items()};actual=evaluate(physical,xs[0][:128].float()).double()@ru.double().T/scale;expected=evaluate(program,xs[0][:128]);drift=float((actual-expected).norm()/expected.norm());path=P/'QUARTIC_ENSEMBLE_FIT_V1.pt';torch.save(archive,path)
 a,b=results['ensemble'],results['half'];pred=dict(pred_a_integrity=replay<1e-4 and drift<1e-4 and math.isfinite(info['objective']),pred_b_transfer=all(a[panel]['response_errors'][str(s)]<=factor*b[panel]['response_errors'][str(s)] for panel,factor in [(0,.8),(1,.9)] for s in HELD),pred_c_values=a[1]['value_error']<=1.1*b[1]['value_error'])
 result=dict(predictions=pred,results=results,training=info,target_replay=replay,export_replay=drift,program_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seconds=time.monotonic()-start,scope='Fixedstates/width/initialization/optimizersteps,fourdonorfamilies; moreconstraints/perstepcompute. Openedpanel/heldpairdiagnostics, not nativeendpoint orsemanticadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
