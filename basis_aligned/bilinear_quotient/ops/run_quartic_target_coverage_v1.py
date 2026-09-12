#!/usr/bin/env python3
# BQGATE:0bodyforwards,0textsequences;16384independent synthetic coefficient probes,no fit.
"""pred_a identities<=1e-8 and estimatednormrelativeSE<=.05; pred_b exactprogramcapture>=.10;
pred_c projectionidentity/distributionagreement within3estimatedSE. Two frozen output directions only.
"""
import os,sys,json,hashlib,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path.insert(0,str(P))
import torch
from composed_quartic_contraction_v1 import contract
from coupled_quartic_writer_v1 import gram,features
STEM='QUARTIC_TARGET_COVERAGE_V1'
def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for v in iter(lambda:f.read(8<<20),b''):h.update(v)
 return h.hexdigest()
def program_contract(x,b,n,mix):
 reads=torch.einsum('nsd,kdr->nskr',x,b);pair=torch.triu_indices(4,4,offset=1,device=x.device)
 q=(reads[:,pair[0]]*reads[:,pair[1]]*n[None,None,:,:]).sum(-1)
 phi=(q[:,0]*q[:,5]+q[:,1]*q[:,4]+q[:,2]*q[:,3])/3
 return phi@mix
def rel(a,b):return float((a-b).norm()/b.norm().clamp_min(1e-30))
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(gpu_accessed=False,body_forwards=0,text_sequences=0,synthetic_probes=16384,fitting=False)));return
 out=P/(STEM+'_RESULT.json');assert not out.exists();signal.alarm(120)
 torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.backends.cuda.matmul.allow_tf32=False;tic=time.perf_counter()
 state=torch.load(next(k for k in binding if k.endswith('/pytorch_model.bin')),weights_only=True,mmap=True)
 p=torch.load(P/'COUPLED_QUARTIC_NONLINEAR_V2_PROGRAM.pt',weights_only=True);b=p['input_readers'].cuda();n=p['inner_weights'].cuda();mix=p['mixing'].cuda();writer=p['output_writers'].cuda()
 u=state['lm_head.weight'].double().cuda();mean=u.mean(0);metric=u.T@u-len(u)*torch.outer(mean,mean);del u
 w=[state[f'transformer.h.{layer}.mlp.{name}.weight'].double().cuda() for layer in (16,17) for name in ('Left','Right','Down')];w[-1]=(metric@writer).T@w[-1];scale=float(state['transformer.h.17.lambdas'][0])
 k=gram(b,n);captured=float((mix*(k@mix)).sum())
 torch.manual_seed(91750);z=torch.randn(8,1152,device='cuda');x=z[:,None,:].expand(-1,4,-1)
 pc=rel(program_contract(x,b,n,mix),features(b,n,z)@mix)
 producer=((z@w[0].T)*(z@w[1].T))@w[2].T*scale;direct=((producer@w[3].T)*(producer@w[4].T))@w[5].T
 nc=rel(contract(x,*w,scale),direct);reports=[]
 for dist,seed in [('gaussian',91751),('rademacher',91752)]:
  torch.manual_seed(seed);norms=[];deltas=[];residuals=[]
  for start in range(0,8192,64):
   x=torch.randn(64,4,1152,device='cuda') if dist=='gaussian' else (torch.randint(0,2,(64,4,1152),device='cuda').double()*2-1)
   y=contract(x,*w,scale);approx=program_contract(x,b,n,mix);target=y.square().sum(-1);residual=(y-approx).square().sum(-1)
   norms.append(target);residuals.append(residual);deltas.append(residual-target+captured)
  values=torch.cat(norms);difference=torch.cat(deltas);mean=float(values.mean());se=float(values.std(unbiased=True)/len(values)**.5);dse=float(difference.std(unbiased=True)/len(values)**.5)
  reports.append(dict(distribution=dist,seed=seed,probes=len(values),target_norm2_estimate=mean,estimated_standard_error=se,relative_standard_error=se/mean,exact_program_coefficient_energy=captured,capture_fraction_estimate=captured/mean,residual_norm2_estimate=float(torch.cat(residuals).mean()),projection_identity_z=float(difference.mean().abs())/max(dse,1e-30)))
 discrepancy=abs(reports[0]['target_norm2_estimate']-reports[1]['target_norm2_estimate'])/(sum(r['estimated_standard_error']**2 for r in reports)**.5)
 result={'pred_a':max(pc,nc)<=1e-8 and all(r['relative_standard_error']<=.05 for r in reports),'pred_b':all(r['capture_fraction_estimate']>=.1 for r in reports),'pred_c':discrepancy<=3 and all(r['projection_identity_z']<=3 for r in reports),'native_diagonal_error':nc,'program_diagonal_error':pc,'reports':reports,'distribution_discrepancy_z':discrepancy,'wall_seconds':time.perf_counter()-tic,'scope':'Full-input frozen two-output coefficient target, not whole-model coverage. Empirical uncertainty, no textfit.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
if __name__=='__main__':main()
