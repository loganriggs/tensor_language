#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_exact pred_b_native64 pred_c_transfer
"""MIDPOINT_NATIVE_PLAN_V1. pred_a_exact midpoint replay<1e-5/fullbasis<1e-10/count80;
pred_b_native64 held error<.5 both; pred_c_transfer held<=1.5cal.
Null no transferable low output rank; price80native forwards, no fitted input circuit.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match';BQ=ROOT/'basis_aligned/bilinear_quotient'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=80,ranks=[4,16,32,64,128,256,512,1024,1152])));return
 import torch
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from native_feature_capture import capture
 from disk_guard import guard_torch_save
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'MIDPOINT_NATIVE_V1.json';assert not out.exists();start=time.perf_counter();model=Bilin18TorchBackend.load('cuda').model.float();b16=model.transformer.h[16];b17=model.transformer.h[17]
 _,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');L=b17.mlp.Left.weight.double();R=b17.mlp.Right.weight.double();C=ru@b17.mlp.Down.weight.double()
 polynomial=lambda x:((x@L.T)*(x@R.T))@C.T
 mixed=lambda n,m:((n@L.T)*(m@R.T)+(m@L.T)*(n@R.T))@C.T
 panels={'calibration':torch.load(BQ/'.rowcache/fineweb_n96_skip1200.pt',weights_only=True)[:32,:65],'fineweb':torch.load(P/'SELECTIVE_CONFIRMATION_FINEWEB_V1.pt',weights_only=True),'code':torch.load(P/'SELECTIVE_CONFIRMATION_CODE_V1.pt',weights_only=True)}
 stats={};checks=[];calls=0;ranks=[4,16,32,64,128,256,512,1024,1152]
 for name,ids in panels.items():
  length=64 if name=='calibration' else 256;ids=ids[:,:length+1];sums={k:torch.zeros(1152,dtype=torch.float64,device='cuda') for k in ['native','shifted']};grams={k:torch.zeros(1152,1152,dtype=torch.float64,device='cuda') for k in sums};count=0
  for row in ids:
   c=capture(model,row[:length].cuda()[None]);h=c['h17'].double().flatten(0,1);m=(b17.lambdas[0]*(c['m16']-b16.mlp.Down_bias)).double().flatten(0,1);n=h-m/2;s=(h.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();n=n/s;m=m/s;h=h/s;y=mixed(n,m);direct=polynomial(h)-polynomial(h-m);checks.append(float((y-direct).norm()/direct.norm()));values={'native':y,'shifted':mixed(n,m.roll(1,0))}
   for kind,y in values.items():sums[kind]+=y.sum(0);grams[kind]+=y.T@y
   count+=length;calls+=1
  stats[name]={k:dict(mean=sums[k]/count,second=grams[k]/count) for k in sums};stats[name]['rows']=count;stats[name]['token_sha256']=hashlib.sha256(ids.numpy().tobytes()).hexdigest();print(name,'captured',flush=True)
 cal=stats['calibration']['native'];mu=cal['mean'];_,basis=torch.linalg.eigh(cal['second']);basis=basis.flip(1);_,cbasis=torch.linalg.eigh(cal['second']-mu[:,None]*mu[None,:]);cbasis=cbasis.flip(1)
 results={};full_errors=[]
 for name,st in stats.items():
  results[name]={}
  for kind in ['native','shifted']:
   gram=st[kind]['second'];mean=st[kind]['mean'];center=gram-mean[:,None]*mu[None,:]-mu[:,None]*mean[None,:]+mu[:,None]*mu[None,:]
   energies=(basis*(gram@basis)).sum(0).cumsum(0);ce=(cbasis*(center@cbasis)).sum(0).cumsum(0);total=gram.trace();ctotal=center.trace()
   full_errors.append(float((total-energies[-1]).abs()/total))
   results[name][kind]=dict(relative_error={str(r):float(((total-energies[r-1]).clamp_min(0)/total).sqrt()) for r in ranks},affine_relative_error={str(r):float(((ctotal-ce[r-1]).clamp_min(0)/total).sqrt()) for r in ranks},constant_mean_relative_error=float((ctotal/total).sqrt()),target_energy=float(total))
 calerr=results['calibration']['native']['relative_error']['64'];pred=dict(pred_a_exact=max(checks)<1e-5 and max(full_errors)<1e-10 and calls==80,pred_b_native64=all(results[d]['native']['relative_error']['64']<.5 for d in ['fineweb','code']),pred_c_transfer=all(results[d]['native']['relative_error']['64']<=1.5*calerr for d in ['fineweb','code']))
 result=dict(predictions=pred,results=results,replay_max=max(checks),fullbasis_energy_residual=max(full_errors),calls=calls,panels={name:{k:st[k] for k in ['rows','token_sha256']} for name,st in stats.items()},seconds=time.perf_counter()-start,scope='Frozen calibration output basis. Exact paired native target, explicit last-MLP denominator, before final normalization/softcap. Reused diagnostics, not circuit or behavioral identification.')
 guard_torch_save(dict(stats={name:{k:({a:b.cpu() for a,b in v.items()} if isinstance(v,dict) else v) for k,v in st.items()} for name,st in stats.items()},basis=basis.cpu(),centered_basis=cbasis.cpu()),str(P/'MIDPOINT_NATIVE_V1.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
