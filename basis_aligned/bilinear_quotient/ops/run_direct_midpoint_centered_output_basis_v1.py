#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_context pred_c_source
"""Original centered-operator output basis at matched width; oracle native screen.
pred_a_instrument Cartesian covariance toy and eigenspace checks <1e-8.
pred_b_context centered256 linear context projection error<paired256 both domains.
pred_c_source centered256 source projection error<=1.1paired256 both domains.
48 native captures plus weight contractions; no heldout fitting.
"""
import os,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,widths=[256,512],metrics=['centered','isotropic'],oracle_only=True)));return
 import torch
 sys.path.insert(0,str(P))
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False
 def covariance(L,R,C,Mn,Mm):
  H=(L@Mn@L.T)*(R@Mm@R.T)+(R@Mn@R.T)*(L@Mm@L.T)+(L@Mn@R.T)*(R@Mm@L.T)+(R@Mn@L.T)*(L@Mm@R.T)
  return C@H@C.T
 g=torch.Generator().manual_seed(26218);rand=lambda *s:torch.randn(*s,generator=g,dtype=torch.float64)
 n,m,L,R,C=rand(7,4),rand(9,4),rand(5,4),rand(5,4),rand(3,5);n-=n.mean(0);m-=m.mean(0)
 y=(((n@L.T)[:,None,:]*(m@R.T)[None,:,:])+((n@R.T)[:,None,:]*(m@L.T)[None,:,:])).reshape(-1,5)@C.T
 direct=y.T@y/len(y);analytic=covariance(L,R,C,n.T@n/len(n),m.T@m/len(m));toy=float((analytic-direct).norm()/direct.norm());assert toy<1e-12
 model=Bilin18TorchBackend.load('cuda').model.float();_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');block=model.transformer.h[17];L=block.mlp.Left.weight.double();R=block.mlp.Right.weight.double();C=ru@block.mlp.Down.weight.double()
 rows=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).cuda().double();m=rows['m'].flatten(0,1).cuda().double();n-=n.mean(0);m-=m.mean(0)
 S=torch.load(P/'MIDPOINT_CENTERED_V1.pt',weights_only=True)['metric_sqrt'].cuda().double();old=torch.load(P/'MIDPOINT_COVERAGE_256_R4_V1.pt',weights_only=True)
 programs={'paired256':dict(span_readers=old['scalar_readers'],span_writers=old['reduced_writers'])};checks=[toy];spectra=[]
 for metric,Mn,Mm in [('centered',n.T@n/len(n),m.T@m/len(m)),('isotropic',torch.eye(1152,device='cuda',dtype=torch.float64),torch.eye(1152,device='cuda',dtype=torch.float64))]:
  cov=covariance(L,R,S@C,Mn,Mm);ev,V=torch.linalg.eigh((cov+cov.T)/2);ev=ev.flip(0).clamp_min(0);V=V.flip(1)
  for width in ([256,512] if metric=='centered' else [256]):
   Q=V[:,:width];checks.append(float((Q.T@Q-torch.eye(width,device='cuda',dtype=torch.float64)).abs().max()));programs[f'{metric}{width}']=dict(span_readers=(S@Q).cpu(),span_writers=torch.linalg.solve(S,Q).cpu());spectra.append(dict(metric=metric,width=width,fit_metric_projection_error=float((ev[width:].sum()/ev.sum()).sqrt())))
 artifact=P/'MIDPOINT_CENTERED_OUTPUT_BASES_V1.pt';assert not artifact.exists();torch.save(programs,artifact)
 del model,L,R,C,n,m,S,cov,ev,V;torch.cuda.empty_cache()
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE='MIDPOINT_CENTERED_CORRECTION_GRAPHS_V1.pt';native.SOURCE_CONTEXT_FAMILIES=True;native.EXTRA_ORACLE_FILE=artifact.name;native.OUTPUT_NAME='MIDPOINT_CENTERED_OUTPUT_BASIS_RAW_V1.json';native.main()
 raw=json.loads((P/native.OUTPUT_NAME).read_text());records=[]
 for d in ['fineweb','code']:
  for f in ['source_only','context_only']:
   for name in programs:
    rr=[v for v in raw['records'] if v['domain']==d and v['family']==f and v['candidate']=='oracle_'+name]
    records.append(dict(domain=d,family=f,basis=name,linear_error=(sum(v['linear_error_energy'] for v in rr)/sum(v['linear_reference_energy'] for v in rr))**.5,native_error=(sum(v['centered_effect_error_energy'] for v in rr)/sum(v['native_centered_effect_energy'] for v in rr))**.5))
 value=lambda d,f,b:next(v['linear_error'] for v in records if v['domain']==d and v['family']==f and v['basis']==b)
 pred=dict(pred_a_instrument=max(checks)<1e-8 and raw['predictions']['pred_a_instrument'],pred_b_context=all(value(d,'context_only','centered256')<value(d,'context_only','paired256') for d in ['fineweb','code']),pred_c_source=all(value(d,'source_only','centered256')<=1.1*value(d,'source_only','paired256') for d in ['fineweb','code']))
 out=P/'MIDPOINT_CENTERED_OUTPUT_BASIS_V1.json';assert not out.exists();out.write_text(json.dumps(dict(predictions=pred,checks=checks,spectra=spectra,records=records,scope='Output subspaces from exact original weights with centered calibration marginal covariances or identity. Paired256 original calibration-output basis control. Native oracle projections use teacher effects, not compressed executable computations. Reused panels; no held fitting.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
