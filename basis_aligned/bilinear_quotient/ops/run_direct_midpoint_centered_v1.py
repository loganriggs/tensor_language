#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_metric pred_b_centered64 pred_c_frozen64
"""Center vocabulary effects before final nonlinearities using exact reduced metric.
pred_a_metric native QR center replay<1e-10, full basis energy residual<1e-10;
pred_b_centered64 refit centered basis held error<.2 both;
pred_c_frozen64 original basis centered held error<.2 both.
Null raw output accuracy is predominantly common-logit energy.
Price zero native forwards, checkpoint QR +1152-square moment transforms.
No fitting held data; native midpoint panels reused; no semantic claim.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(native_forwards=0,output_dimension=1152)));return
 import torch
 from circuit_fast_screen_producer import Bilin18TorchBackend
 from disk_guard import guard_torch_save
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 out=P/'MIDPOINT_CENTERED_V1.json';assert not out.exists();start=time.perf_counter();model=Bilin18TorchBackend.load('cuda').model.float();Q,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');a=Q.sum(0)/Q.shape[0]**.5;M=torch.eye(1152,dtype=torch.float64,device='cuda')-a[:,None]*a[None,:];ev,E=torch.linalg.eigh(M);assert float(ev.min())>-1e-10;S=(E*ev.clamp_min(0).sqrt()[None,:])@E.T
 gen=torch.Generator(device='cuda').manual_seed(261044);y=torch.randn(3,1152,dtype=torch.float64,device='cuda',generator=gen);v=y@Q.T;vc=v-v.mean(-1,keepdim=True);replay=float(((y@M*y).sum()-vc.square().sum()).abs()/vc.square().sum());artifact=torch.load(P/'MIDPOINT_NATIVE_V1.pt',weights_only=True);stats=artifact['stats'];B=artifact['basis'].cuda();cal=stats['calibration']['native']['second'].cuda();cg=S@cal@S;_,W=torch.linalg.eigh(cg);W=W.flip(1);ranks=[4,16,32,64,128,256,512,1024,1152];results={};checks=[]
 for name,st in stats.items():
  G=st['native']['second'].cuda();energy=G.trace();centered=(M*G.T).sum();SGS=S@G@S;kept=(W*(SGS@W)).sum(0).cumsum(0);original={}
  for r in ranks:
   U=B[:,:r];PU=U@U.T;errorG=G-PU@G-G@PU+PU@G@PU;error=(M*errorG.T).sum();original[str(r)]=float((error.clamp_min(0)/centered).sqrt())
  results[name]=dict(common_energy_fraction=float((energy-centered)/energy),centered_basis_error={str(r):float(((centered-kept[r-1]).clamp_min(0)/centered).sqrt()) for r in ranks},original_basis_centered_error=original)
  checks.append(float((centered-kept[-1]).abs()/centered))
 pred=dict(pred_a_metric=replay<1e-10 and max(checks)<1e-10,pred_b_centered64=all(results[d]['centered_basis_error']['64']<.2 for d in ['fineweb','code']),pred_c_frozen64=all(results[d]['original_basis_centered_error']['64']<.2 for d in ['fineweb','code']))
 result=dict(predictions=pred,results=results,metric_replay=replay,fullbasis_relative_energy_residual=max(checks),minimum_metric_eigenvalue=float(ev.min()),seconds=time.perf_counter()-start,scope='Vocabulary centering before final RMS/softcap, so this is not centered final-logit intervention error. Centered basis fits calibration only; raw basis also audited without refit.')
 guard_torch_save(dict(metric=M.cpu(),metric_sqrt=S.cpu(),centered_basis=W.cpu(),common_direction=a.cpu()),str(P/'MIDPOINT_CENTERED_V1.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
