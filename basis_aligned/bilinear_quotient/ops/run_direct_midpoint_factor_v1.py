#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact pred_b_weighted16 pred_c_gain
"""MIDPOINT_FACTOR_PLAN_V1: a) K replay<1e-10/fullSVD<1e-8/toy<1e-12;
b) weighted rank16 all scalar baseline-relative error<.5 held both;
c) weighted rank16 aggregate MSE <=.8 isotropic both held.
Null no cheap scalar forms or marginal metric misses paired inputs.
Price80forwards,8matrixSVD1152,4rproducts/9216rinputcoefficients per arm.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(forwards=80,scalars=4,ranks=[1,4,16,64],metrics=['isotropic','separable_moment'])));return
 import torch
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from weighted_bilinear_svd import roots,decompose,toy_check
 from disk_guard import guard_torch_save
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'MIDPOINT_FACTOR_V1.json';assert not out.exists();start=time.perf_counter();toy=toy_check();model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17];_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();C=ru@b17.mlp.Down.weight.double()
 artifact=torch.load(P/'MIDPOINT_NATIVE_V1.pt',weights_only=True);audit=torch.load(P/'MIDPOINT_CENTERED_V1.pt',weights_only=True);S=audit['metric_sqrt'].cuda();stats=artifact['stats']['calibration']['native'];mu=stats['mean'].cuda();cov=stats['second'].cuda()-mu[:,None]*mu[None,:];_,W=torch.linalg.eigh(S@cov@S);W=W.flip(1)[:,:4];readers=S@W;offset=mu@readers;channel=readers.T@C;K=torch.stack([L.T@(c[:,None]*R)+R.T@(c[:,None]*L) for c in channel])
 panels={'calibration':torch.load(BQ/'.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:32,:65],'fineweb':torch.load(P/'SELECTIVE_CONFIRMATION_FINEWEB_V1.pt',weights_only=True),'code':torch.load(P/'SELECTIVE_CONFIRMATION_CODE_V1.pt',weights_only=True)};data={};checks=[];hashes={}
 for name,ids in panels.items():
  length=64 if name=='calibration' else 256;ids=ids[:,:length+1];rows=[];hashes[name]=hashlib.sha256(ids.numpy().tobytes()).hexdigest();assert hashes[name]==artifact['stats'][name]['token_sha256']
  for row in ids:
   c=capture(model,row[:length].cuda()[None]);h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);scale=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=(h-m/2)/scale;m=m/scale;y=((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@channel.T;direct=torch.einsum('bi,gij,bj->bg',n,K,m);checks.append(float((direct-y).norm()/y.norm()));rows.append((n,m,y-offset))
  data[name]=tuple(torch.cat([r[i] for r in rows]) for i in range(3));print(name,'captured',flush=True)
 n,m,_=data['calibration'];left=roots(n.T@n/len(n));right=roots(m.T@m/len(m));eye=torch.eye(1152,dtype=torch.float64,device='cuda');programs={};fullchecks=[];spectra={}
 for metric in ['isotropic','separable_moment']:
  fac=[decompose(k,(eye,eye) if metric=='isotropic' else left,(eye,eye) if metric=='isotropic' else right) for k in K];fullchecks.extend(float((a@b.T-k).norm()/k.norm()) for (a,b,s),k in zip(fac,K));spectra[metric]=[s.cpu().tolist() for a,b,s in fac]
  for rank in [1,4,16,64]:programs[f'{metric}_{rank}']=dict(A=torch.stack([a[:,:rank] for a,b,s in fac]),B=torch.stack([b[:,:rank] for a,b,s in fac]),offset=offset,products=4*rank,input_coefficients=2*1152*4*rank)
 results={}
 for name,(n,m,y) in data.items():
  results[name]={}
  for key,prog in programs.items():
   pred=(torch.einsum('bi,gir->bgr',n,prog['A'])*torch.einsum('bi,gir->bgr',m,prog['B'])).sum(-1)-offset;error=(pred-y).square().sum(0);energy=y.square().sum(0);results[name][key]=dict(relative_error=(error/energy).sqrt().cpu().tolist(),aggregate_relative_error=float((error.sum()/energy.sum()).sqrt()),squared_error=float(error.sum()),baseline_energy=float(energy.sum()))
 pred=dict(pred_a_exact=max(checks)<1e-10 and max(fullchecks)<1e-8 and max(toy.values())<1e-12,pred_b_weighted16=all(max(results[d]['separable_moment_16']['relative_error'])<.5 for d in ['fineweb','code']),pred_c_gain=all(results[d]['separable_moment_16']['squared_error']<=.8*results[d]['isotropic_16']['squared_error'] for d in ['fineweb','code']))
 saved={k:{a:(b.float().cpu() if torch.is_tensor(b) else b) for a,b in v.items()} for k,v in programs.items()};guard_torch_save(dict(programs=saved,scalar_readers=readers.cpu(),scalar_mean=offset.cpu(),reduced_writers=torch.linalg.solve(S,W).cpu(),K=K.float().cpu(),scope='Inputs normalized midpoint and previous residual source; output scalars centered on calibration means. Native upstream projection/normalization explicit.'),str(P/'MIDPOINT_FACTOR_PROGRAMS_V1.pt'))
 result=dict(predictions=pred,results=results,toy=toy,exact_replay=max(checks),full_svd_replay=max(fullchecks),token_hashes=hashes,spectra=spectra,seconds=time.perf_counter()-start,scope='Four independent output-shared bilinear forms. Native paired variation metric, no cross-scalar reuse or behavioral validation yet.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='spectra'},indent=2))
if __name__=='__main__':main()
