#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_transfer pred_c_preservation
"""Two100-step Adam feature fits, fixed32x4dictionarysize/16writers.
Export<1e-4; sensitive root1<=.8fixed,meanroots<=.85fixed;values<=1.1fixed.
Both<=384products/330240coeff. Null: movingfeaturesoverfits ordoesnottransfer.
"""
import os,sys,time,json,math,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def main():
 import torch
 sys.path.insert(0,str(P));from sensitive_root_features import fit,controls
 from empirical_quartic_dictionary import features
 from audit_root_feature_conditions import root_features
 from paired_root_compiler import cast,price
 from run_sensitive_root_fit_v2 import compile_forms
 torch.set_num_threads(2)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checks=controls();torch.manual_seed(511);x=torch.randn(19,1152,dtype=torch.float64);y=torch.randn(19,16,dtype=x.dtype);w=torch.rand_like(y)+.1;initial=(torch.randn(3,2,1152,dtype=x.dtype)/34,torch.randn(3,2,1152,dtype=x.dtype)/34);info,p=fit(x,y,w,initial,steps=2,rate=.0001);base=dict(U=p['U'],V=p['V'],writer=torch.randn(1152,16,dtype=x.dtype));program,_=compile_forms(base,p['coefficients']);truth=features(x,p['U'],p['V'])@p['coefficients'].T;err=float((root_features(program,x)-truth).norm()/truth.norm());assert err<1e-9;print(json.dumps(dict(controls=checks,train_compile_smoke=err)));return
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'SENSITIVE_ROOT_FEATURE_FIT_V1.json';assert not out.exists();data=torch.load(P/'SENSITIVE_ROOT_CALIBRATION_V2.pt',weights_only=True)['panels'];panels=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'];extra=torch.load(P/'QUARTIC_ADDITIONAL_STATES_V1.pt',weights_only=True)['rows'];xs=[torch.cat([panels[0]['rows'],extra]).cuda().double(),panels[1]['rows'].cuda().double()];ys=[r['target'].cuda() for r in data];ws=[r['weight'].cuda() for r in data]
 base=cast(torch.load(P/'EXPANDED_ROOT_EMPIRICAL_V1.pt',weights_only=True),torch.float64);initial=(base['U'].cuda(),base['V'].cuda());results={};histories={};prices={};diags={};hashes={};drifts=[]
 def move(x):
  if torch.is_tensor(x):return x.cuda()
  if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
  if isinstance(x,list):return [move(v) for v in x]
  return x
 def assess(pred,y,w):
  e=(pred-y).square();t=y.square();r=((e*w).sum(0)/(t*w).sum(0)).sqrt();return dict(value_error=float(e.sum().sqrt()/t.sum().sqrt()),weighted_root_errors=r.tolist(),mean_root_weighted_error=float(r.mean()))
 for arm in ['uniform','sensitive']:
  info,p=fit(xs[0],ys[0],torch.ones_like(ws[0]) if arm=='uniform' else ws[0],initial,steps=100,rate=.03*math.sqrt(4/1152));histories[arm]=info
  with torch.no_grad():
   preds=[features(x,p['U'],p['V'])@p['coefficients'].T for x in xs];results[arm]=[assess(pred,y,w) for pred,y,w in zip(preds,ys,ws)];source=dict(U=p['U'],V=p['V'],writer=base['writer']);program,diags[arm]=compile_forms(source,p['coefficients']);program=cast(program,torch.float32);prices[arm]=price(program);path=P/f'SENSITIVE_ROOT_FEATURE_{arm.upper()}_V1.pt';torch.save(program,path);hashes[arm]=hashlib.sha256(path.read_bytes()).hexdigest();actual=move(program)
   for x,pred in zip(xs,preds):drifts.append(float((root_features(actual,x[:128].float()).double()-pred[:128]).norm()/pred[:128].norm()))
  print(json.dumps(dict(arm=arm,training=info,results=results[arm],price=prices[arm])),flush=True)
 fixed=json.loads((P/'SENSITIVE_ROOT_FIT_V2.json').read_text())['results'];a=results['sensitive'][1];b=fixed['sensitive'][1];pred=dict(pred_a_integrity=max(drifts)<1e-4 and all(math.isfinite(v['objective']) for v in histories.values()),pred_b_transfer=a['weighted_root_errors'][1]<=.8*b['weighted_root_errors'][1] and a['mean_root_weighted_error']<=.85*b['mean_root_weighted_error'],pred_c_preservation=a['value_error']<=1.1*b['value_error'] and all(v['products']<=384 and v['stored_coefficients']<=330240 for v in prices.values()))
 result=dict(predictions=pred,results=results,fixed_baselines=fixed,histories=histories,prices=prices,compiler_diagnostics=diags,program_sha256=hashes,maximum_export_replay=max(drifts),seconds=time.monotonic()-start)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['compiler_diagnostics','fixed_baselines']},indent=2))
if __name__=='__main__':main()
