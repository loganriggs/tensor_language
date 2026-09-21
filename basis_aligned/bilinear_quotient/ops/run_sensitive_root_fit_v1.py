#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_integrity pred_b_transfer pred_c_preservation
"""Fixed16writers/32quadraticfeatures, native-root weighted readout fit.
Capture/export<1e-4; evalroot1weighted<=.8uniform,meanrooterror<=.9uniform.
Value<=1.1uniform and<=384products/330240coeff both. Ridge1e-6.
Null: sensitivity weighting does not transfer. No semanticadoption.
"""
import os,sys,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'

def compile_forms(base,coeff):
 import torch
 from paired_root_compiler import compile_program
 coeff=coeff.cpu().double();m=base['U'].shape[0];i,j=torch.triu_indices(m,m);forms=coeff.new_zeros(len(coeff),m,m);forms[:,i,j]=coeff/torch.where(i==j,1.,2.);forms[:,j,i]=forms[:,i,j];e,v=torch.linalg.eigh(forms)
 source={k:base[k].cpu().double() for k in ['U','V','writer']};source.update(root_eigenvalues=e,root_eigenvectors=v)
 return compile_program(source,allow_rotations=True)

def main():
 import torch
 sys.path.insert(0,str(P));from sensitive_root_readout import sensitivity,fit_roots,controls
 from empirical_quartic_dictionary import features
 from audit_root_feature_conditions import root_features
 from paired_root_compiler import cast,price
 torch.set_num_threads(2);torch.set_grad_enabled(False)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  checks=controls();torch.manual_seed(72);x=torch.randn(9,1152,dtype=torch.float64);base=dict(U=torch.randn(32,4,1152,dtype=x.dtype),V=torch.randn(32,4,1152,dtype=x.dtype),writer=torch.randn(1152,16,dtype=x.dtype));coeff=torch.randn(16,528,dtype=x.dtype);p,diag=compile_forms(base,coeff);truth=features(x,base['U'],base['V'])@coeff.T;error=float((root_features(p,x)-truth).norm()/truth.norm());assert error<1e-9;print(json.dumps(dict(controls=checks,fullshape_replay=error,price=price(p))));return
 from native_feature_capture import capture
 from native_quartic_branch import pure_branch
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.backends.cuda.matmul.allow_tf32=False;start=time.monotonic();out=P/'SENSITIVE_ROOT_FIT_V1.json';assert not out.exists()
 path=P/'EXPANDED_ROOT_EMPIRICAL_V1.pt';sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha=='f50ab7fe62295fd338e883c486ca1d77d3dbf9d3d21883f8b7b95393bf6fb2df'
 base=cast(torch.load(path,weights_only=True),torch.float64);model=Bilin18TorchBackend.load('cuda').model.float();b16,b17=model.transformer.h[16],model.transformer.h[17];W=base['writer'].cuda().float();uv=model.lm_head.weight@W;gram=uv.T@uv;readers=model.lm_head.weight.T@uv/gram.diag();calcache=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels'][0]['rows'][:256].cuda()
 datasets=[];replay=0.;hashes={}
 for name,file,n in [('calibration','fineweb_n96_skip1200.pt',96),('evaluation','fineweb_n192_skip7000.pt',32)]:
  ids=torch.load(ROOT/'basis_aligned/bilinear_quotient/.rowcache'/file,weights_only=True)[:n,:64];hashes[name]=hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest();xx=[];yy=[];ww=[]
  for index in range(0,n,4):
   c=capture(model,ids[index:index+4].cuda());x,m,h,final=[c[k].flatten(0,1) for k in ['x16','m16','h17','final']]
   if name=='calibration' and index==0:replay=float((x-calcache).norm()/calcache.norm())
   l,r,d=[getattr(b17.mlp,k).weight for k in ['Left','Right','Down']];target=pure_branch(m,b16.mlp.Down_bias,b17.lambdas[0],l,r,d)@readers;den=h.square().mean(-1,keepdim=True)+torch.finfo(h.dtype).eps
   # Double for the three-term squared-norm contraction to protect cancellation.
   weights=sensitivity(final.double(),W.double(),den.double(),model.lm_head.weight.double())
   xx.append(x.double());yy.append(target.double());ww.append(weights)
  datasets.append(dict(x=torch.cat(xx),target=torch.cat(yy),weight=torch.cat(ww)))
  print(json.dumps(dict(captured=name,rows=n*64)),flush=True)
 del model
 U,V=base['U'].cuda(),base['V'].cuda();phis=[features(ds['x'],U,V) for ds in datasets];coeffs={arm:fit_roots(phis[0],datasets[0]['target'],torch.ones_like(datasets[0]['weight']) if arm=='uniform' else datasets[0]['weight']) for arm in ['uniform','sensitive']}
 def measure(pred,ds):
  e=(pred-ds['target']).square();t=ds['target'].square();w=ds['weight'];roots=((e*w).sum(0)/(t*w).sum(0)).sqrt()
  return dict(value_error=float(e.sum().sqrt()/t.sum().sqrt()),weighted_root_errors=roots.tolist(),mean_root_weighted_error=float(roots.mean()))
 results={};prices={};diagnostics={};exports={};drifts=[]
 def move(v):
  if torch.is_tensor(v):return v.cuda()
  if isinstance(v,dict):return {k:move(t) for k,t in v.items()}
  if isinstance(v,list):return [move(t) for t in v]
  return v
 inherited=move(base);results['inherited']=[measure(root_features(inherited,ds['x']),ds) for ds in datasets]
 for arm,coeff in coeffs.items():
  results[arm]=[measure(phi@coeff.T,ds) for phi,ds in zip(phis,datasets)];program,diag=compile_forms(base,coeff);program=cast(program,torch.float32);prices[arm]=price(program);diagnostics[arm]=diag;program_path=P/f'SENSITIVE_ROOT_{arm.upper()}_V1.pt';torch.save(program,program_path);exports[arm]=hashlib.sha256(program_path.read_bytes()).hexdigest();physical=move(program)
  for phi,ds in zip(phis,datasets):
   expected=phi[:128]@coeff.T;actual=root_features(physical,ds['x'][:128].float()).double();drifts.append(float((actual-expected).norm()/expected.norm()))
 torch.save(dict(panels=[{k:v.cpu() for k,v in ds.items() if k!='x'} for ds in datasets],token_sha256=hashes,source_program_sha256=sha),P/'SENSITIVE_ROOT_CALIBRATION_V1.pt')
 a,b=results['sensitive'][1],results['uniform'][1];pred=dict(pred_a_integrity=replay<1e-4 and max(drifts)<1e-4,pred_b_transfer=a['weighted_root_errors'][1]<=.8*b['weighted_root_errors'][1] and a['mean_root_weighted_error']<=.9*b['mean_root_weighted_error'],pred_c_preservation=a['value_error']<=1.1*b['value_error'] and all(v['products']<=384 and v['stored_coefficients']<=330240 for v in prices.values()))
 result=dict(predictions=pred,results=results,prices=prices,compiler_diagnostics=diagnostics,capture_replay=replay,maximum_export_replay=max(drifts),program_sha256=exports,token_sha256=hashes,writer_gram_offdiag=float((gram-torch.diag(gram.diag())).abs().max()),seconds=time.monotonic()-start)
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='compiler_diagnostics'},indent=2))
if __name__=='__main__':main()
