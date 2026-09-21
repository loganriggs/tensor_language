#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_rank8 pred_c_weighting
"""Shared midpoint reader graph, ranks4/8/12, coefficient vs native-moment spans.
pred_a_replay fullspan scalar error<1e-10 and explicit graph vs collapsed<1e-10;
pred_b_rank8 weightedrank8 aggregate error<=1.25confirmed andallfeatureerror<.2 bothheld;
pred_c_weighting weightedrank8 squared error<=.8isotropicrank8 bothheld.
Null sharing destroys confirmed functions or marginal reader metric fails to preserve products.
Price80nativeforwards; no feature selection/output changes. Rank8 graph18,688readercoeff/16products,
rank4=9344/rank12=28032 vsconfirmed36,864. Additional4means+4608writers unchanged.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=80,shared_ranks=[4,8,12],products=16,metrics=['isotropic','native_moment'])));return
 import torch
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from weighted_bilinear_svd import roots
 from disk_guard import guard_torch_save
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;out=P/'MIDPOINT_SHARED_FIT_V1.json';assert not out.exists();start=time.perf_counter();model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];export=torch.load(P/'MIDPOINT_EXTRACTED_PROGRAM_V1.pt',weights_only=True);e={k:v.cuda().double() for k,v in export.items()};A=e['A'];B=e['B'];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();C=e['scalar_readers'].T@ru@b17.mlp.Down.weight.double()
 panels={'calibration':torch.load(BQ/'.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:32,:65],'fineweb':torch.load(P/'SELECTIVE_CONFIRMATION_FINEWEB_V1.pt',weights_only=True),'code':torch.load(P/'SELECTIVE_CONFIRMATION_CODE_V1.pt',weights_only=True)};data={};hashes={}
 for name,ids in panels.items():
  length=64 if name=='calibration' else 256;rows=[];hashes[name]=hashlib.sha256(ids[:,:length+1].numpy().tobytes()).hexdigest()
  for row in ids:
   c=capture(model,row[None,:length].cuda());h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-m/2)/scale;m=m/scale;y=((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@C.T-e['offset'];rows.append((n,m,y))
  data[name]=tuple(torch.cat([r[i] for r in rows]) for i in range(3))
 n,m,_=data['calibration'];Sn,In=roots(n.T@n/len(n));Sm,Im=roots(m.T@m/len(m));eye=torch.eye(1152,device='cuda',dtype=torch.float64);programs={};checks=[]
 for metric in ['isotropic','native_moment']:
  sn,invn,sm,invm=(eye,eye,eye,eye) if metric=='isotropic' else (Sn,In,Sm,Im)
  scale=((sm@B).norm(dim=0)/(sn@A).norm(dim=0)).sqrt();aa=A*scale;bb=B/scale;Ua,_,_=torch.linalg.svd(sn@aa,full_matrices=False);Ub,_,_=torch.linalg.svd(sm@bb,full_matrices=False)
  for rank in [4,8,12,16]:
   Pn=invn@Ua[:,:rank];Pm=invm@Ub[:,:rank];Tn=Ua[:,:rank].T@sn@aa;Tm=Ub[:,:rank].T@sm@bb;programs[f'{metric}_{rank}']=dict(Pn=Pn,Pm=Pm,Tn=Tn,Tm=Tm,offset=e['offset'],readout=e['readout'],scalar_readers=e['scalar_readers'],reduced_writers=e['reduced_writers'])
 results={}
 for name,(n,m,y) in data.items():
  baseline=((n@A)*(m@B))@e['readout']-e['offset'];den=y.square().sum(0);result={}
  def measure(pred):
   errors=(pred-y).square().sum(0);return dict(relative_error=(errors/den).sqrt().cpu().tolist(),aggregate_relative_error=float((errors.sum()/den.sum()).sqrt()),squared_error=float(errors.sum()),difference_from_confirmed=float((pred-baseline).norm()/y.norm()))
  result['confirmed']=measure(baseline)
  for key,g in programs.items():
   pred=(((n@g['Pn'])@g['Tn'])*((m@g['Pm'])@g['Tm']))@g['readout']-g['offset'];collapsed=((n@(g['Pn']@g['Tn']))*(m@(g['Pm']@g['Tm'])))@g['readout']-g['offset'];checks.append(float((pred-collapsed).norm()/y.norm()))
   if key.endswith('_16'):checks.append(float((pred-baseline).norm()/y.norm()))
   result[key]=measure(pred)
  results[name]=result
 pred=dict(pred_a_replay=max(checks)<1e-10,pred_b_rank8=all(results[d]['native_moment_8']['aggregate_relative_error']<=1.25*results[d]['confirmed']['aggregate_relative_error'] and max(results[d]['native_moment_8']['relative_error'])<.2 for d in ['fineweb','code']),pred_c_weighting=all(results[d]['native_moment_8']['squared_error']<=.8*results[d]['isotropic_8']['squared_error'] for d in ['fineweb','code']))
 guard_torch_save(dict(programs={k:{a:b.float().cpu() for a,b in v.items()} for k,v in programs.items()},regularized_calibration_roots=dict(n=Sn.cpu(),m=Sm.cpu())),str(P/'MIDPOINT_SHARED_FIT_PROGRAMS_V1.pt'))
 result=dict(predictions=pred,results=results,replay_max=max(checks),token_hashes=hashes,seconds=time.perf_counter()-start,scope='Shared linear input dictionaries, fixed16products and4outputfeatures. Calibration moments only. Reused diagnostic native pairs; no native intervention claim yet.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
