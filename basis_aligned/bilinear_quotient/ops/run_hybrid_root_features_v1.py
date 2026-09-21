#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_global pred_c_text
"""NativeGaussian-only vs lambda1hybrid,100Adamsteps/fixed32x4/16writers.
Export/teacher<1e-4/1e-5; hybridGaussian<=.5oldtextfit;
textroot1/mean<=1.1oldtextfit and384products330240coeffboth.
Null: global/textfitconflict. Gaussianfunctional, notcoefficientmetric.
"""
import os,sys,time,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P));from hybrid_root_metric import combine,controls
 from sensitive_root_features import fit
 from empirical_quartic_dictionary import features
 from paired_root_compiler import cast,price
 from audit_root_feature_conditions import root_features
 from audit_root_matched_reader import CK
 from run_sensitive_root_fit_v2 import compile_forms
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checks=controls();torch.manual_seed(17);x=torch.randn(23,7,dtype=torch.float64);g=torch.randn(17,7,dtype=x.dtype);y=torch.randn(23,4,dtype=x.dtype);z=torch.randn(17,4,dtype=x.dtype);w=torch.rand_like(y)+.1;xx,yy,ww=combine(x,y,w,g,z,1.);initial=(torch.randn(3,2,7,dtype=x.dtype),torch.randn(3,2,7,dtype=x.dtype));info,p=fit(xx,yy,ww,initial,steps=2,rate=.0001);assert math.isfinite(info['objective']);print(json.dumps(dict(mixture_controls=checks,mixed_fit_smoke=info)));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'HYBRID_ROOT_FEATURE_V1.json';assert not out.exists();base=cast(torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True),torch.float64);initial=(base['U'].cuda(),base['V'].cuda())
 data=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)['rows'];tx=[torch.cat([panels[0]['rows'],extra]).cuda().double(),panels[1]['rows'].cuda().double()];ty=[r['target'].cuda() for r in data];tw=[r['weight'].cuda() for r in data]
 s=torch.load(CK,weights_only=True,mmap=True,map_location='cpu');uv=s['lm_head.weight'].cuda().double();uw=uv@base['writer'].cuda();readers=uv.T@uw/(uw.square().sum(0));del uv,uw
 def weight(layer,name):return s[f'transformer.h.{layer}.mlp.{name}.weight'].cuda().double()
 A,B,D=weight(16,'Left'),weight(16,'Right'),weight(16,'Down')*s['transformer.h.17.lambdas'][0].item();L,R,O=weight(17,'Left'),weight(17,'Right'),weight(17,'Down');folded=O.T@readers
 def teacher(x):
  m=((x@A.T)*(x@B.T))@D.T;return ((m@L.T)*(m@R.T))@folded
 gx=[torch.randn(n,1152,generator=torch.Generator().manual_seed(seed),dtype=torch.float64).cuda() for n,seed in [(2048,938),(1024,939)]]
 with torch.no_grad():
  gy=[teacher(x) for x in gx];x=gx[0][:8];m=((x@A.T)*(x@B.T))@D.T;unfold=(((m@L.T)*(m@R.T))@O.T)@readers;teacher_replay=float((teacher(x)-unfold).norm()/unfold.norm())
 def move(x):
  if torch.is_tensor(x):return x.cuda()
  if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
  if isinstance(x,list):return [move(v) for v in x]
  return x
 def assess(h,y,w):
  error=(h-y).square();reference=y.square();r=((error*w).sum(0)/(reference*w).sum(0)).sqrt();return dict(value_error=float(error.sum().sqrt()/reference.sum().sqrt()),weighted_root_errors=r.tolist(),mean_root_weighted_error=float(r.mean()))
 results={};baselines={};histories={};prices={};penalties={};drifts=[];hashes={}
 with torch.no_grad():
  for name,file in [('inherited','EXPANDED_ROOT_EMPIRICAL_V1.pt'),('text_sensitive','SENSITIVE_ROOT_FEATURE_SENSITIVE_V1.pt')]:
   program=move(cast(torch.load(P/file,weights_only=True),torch.float64));baselines[name]=dict(text=[assess(root_features(program,x),y,w) for x,y,w in zip(tx,ty,tw)],gaussian=[assess(root_features(program,x),y,torch.ones_like(y)) for x,y in zip(gx,gy)])
 for arm in ['synthetic','hybrid']:
  if arm=='synthetic':x,y,w=gx[0],gy[0],torch.ones_like(gy[0])
  else:x,y,w=combine(tx[0],ty[0],tw[0],gx[0],gy[0],1.)
  info,p=fit(x,y,w,initial,steps=100,rate=.03*math.sqrt(4/1152));histories[arm]=info
  with torch.no_grad():
   predict=lambda z:features(z,p['U'],p['V'])@p['coefficients'].T
   results[arm]=dict(text=[assess(predict(xx),yy,ww) for xx,yy,ww in zip(tx,ty,tw)],gaussian=[assess(predict(xx),yy,torch.ones_like(yy)) for xx,yy in zip(gx,gy)])
   phi=features(x,p['U'],p['V']);sc=phi.square().mean(0).sqrt();ww=w/w.mean(0,keepdim=True);energy=(ww*y.square()).sum(0);err=(ww*(predict(x)-y).square()).sum(0);pen=len(x)*1e-6*(p['coefficients']*sc).square().sum(1);penalties[arm]=dict(mean_data_loss=float((err/energy).mean()),mean_ridge_loss=float((pen/energy).mean()),ridge_to_data=float((pen/energy).mean()/(err/energy).mean()))
   source=dict(U=p['U'],V=p['V'],writer=base['writer']);program,diagnostic=compile_forms(source,p['coefficients']);program=cast(program,torch.float32);prices[arm]=price(program);path=P/f'HYBRID_ROOT_{arm.upper()}_V1.pt';torch.save(program,path);hashes[arm]=hashlib.sha256(path.read_bytes()).hexdigest();actual=move(program)
   for xx in [tx[1][:128],gx[1][:128]]:drifts.append(float((root_features(actual,xx.float()).double()-predict(xx)).norm()/predict(xx).norm()))
  print(json.dumps(dict(arm=arm,training=info,results=results[arm],ridge=penalties[arm],price=prices[arm])),flush=True)
 a=results['hybrid'];b=baselines['text_sensitive'];pred=dict(pred_a_integrity=teacher_replay<1e-5 and max(drifts)<1e-4 and all(math.isfinite(v['objective']) for v in histories.values()),pred_b_global=a['gaussian'][1]['value_error']<=.5*b['gaussian'][1]['value_error'],pred_c_text=a['text'][1]['weighted_root_errors'][1]<=1.1*b['text'][1]['weighted_root_errors'][1] and a['text'][1]['mean_root_weighted_error']<=1.1*b['text'][1]['mean_root_weighted_error'] and all(v['products']<=384 and v['stored_coefficients']<=330240 for v in prices.values()))
 result=dict(predictions=pred,results=results,baselines=baselines,histories=histories,penalties=penalties,prices=prices,program_sha256=hashes,maximum_export_replay=max(drifts),teacher_replay=teacher_replay,seconds=time.monotonic()-start,seeds=[938,939],scope='Synthetic-onlyobjective retainsdata-informedinit/writers; GaussianfunctionmetricnotcoeffFrobenius; textpanelopened,noadoption.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['baselines','histories']},indent=2))
if __name__=='__main__':main()
