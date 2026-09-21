#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_isotropic pred_c_low_variance
"""Exact coefficient errors of three frozen programs; no fitting/forwards.
Covariance errors reproduce<1e-5. Learned isotropic error>1.25fixed-response;
lowest covariance-eigenvalue quarter carries>50%learned error input-slot energy.
Metrics concern homogeneous quadratic coefficients; affine correction excluded.
"""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 import torch
 sys.path.insert(0,str(P));from quadratic_mode_grams import grams,controls
 torch.set_num_threads(2);check=controls()
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(dict(controls=check,forwards=0,programs=3)));return
 from learned_quadratic_factors import inner
 torch.set_grad_enabled(False);start=time.monotonic();out=P/'FULL_QUADRATIC_GEOMETRY_AUDIT_V1.json';assert not out.exists()
 state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True,mmap=True,map_location='cpu')
 def w(k):return state[k].cuda().double()
 U=w('lm_head.weight');RU=torch.linalg.cholesky(U.T@U).T;del U
 C=RU@w('transformer.h.17.mlp.Down.weight');A=w('transformer.h.17.mlp.Left.weight');B=w('transformer.h.17.mlp.Right.weight');data=torch.load(P/'SHARED_PRODUCT_NATIVE_INPUTS_V1.pt',weights_only=True);x=data['h'].cuda().double();x=x/(x.square().mean(-1,keepdim=True)+torch.finfo(torch.float32).eps).sqrt();x-=x.mean(0);ev,V=torch.linalg.eigh(x.T@x/len(x));floor=1e-8*ev.mean();root=(V*ev.clamp_min(floor).sqrt())@V.T;_,nativeG=grams(C,A,B);den=float(nativeG.trace());native_diag=(V*(nativeG@V)).sum(0);weighted_den=float(inner(C,A@root,B@root,C,A@root,B@root));rows=[]
 names=[('pruned','FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt'),('response_refit','FULL_CHANNEL_RESPONSE_PROGRAM_V1.pt'),('learned','LEARNED_FULL_QUADRATIC_PROGRAM_V1.pt')]
 for name,file in names:
  p={k:v.cuda().double() for k,v in torch.load(P/file,weights_only=True).items()};c=RU@p['writer'];a,b=p['a'],p['b'];cc=torch.cat([C,-c],1);aa,bb=torch.cat([A,a]),torch.cat([B,b]);_,G=grams(cc,aa,bb);energy=float(G.trace());diag=(V*(G@V)).sum(0);weighted=float(inner(cc,aa@root,bb@root,cc,aa@root,bb@root));bands=[]
  for j in range(4):
   sl=slice(j*288,(j+1)*288);e=float(diag[sl].sum());t=float(native_diag[sl].sum());bands.append(dict(ascending_variance_quarter=j,error_energy_fraction=e/energy,teacher_energy_fraction=t/den,band_relative_error=(max(e,0)/t)**.5))
  rows.append(dict(program=name,isotropic_coefficient_error=(max(energy,0)/den)**.5,covariance_coefficient_error=(max(weighted,0)/weighted_den)**.5,bands=bands));print(json.dumps(rows[-1]),flush=True)
 old=json.loads((P/'FULL_CHANNEL_RESPONSE_REFIT_V1.json').read_text());new=json.loads((P/'LEARNED_FULL_QUADRATIC_V1.json').read_text());expected=[old['records'][0]['coefficient_error'],old['records'][2]['coefficient_error'],next(r['coefficient_error'] for r in new['records'] if r['start']==new['selected'])];replay=max(abs(r['covariance_coefficient_error']-v) for r,v in zip(rows,expected));pred=dict(pred_a_replay=replay<1e-5,pred_b_isotropic=rows[2]['isotropic_coefficient_error']>1.25*rows[1]['isotropic_coefficient_error'],pred_c_low_variance=rows[2]['bands'][0]['error_energy_fraction']>.5)
 result=dict(predictions=pred,records=rows,maximum_covariance_replay=replay,covariance_eigenvalue_range=[float(ev.min()),float(ev.max())],floored_eigenvalues=int((ev<floor).sum()),controls=check,seconds=time.monotonic()-start,scope='Exact symmetric quadratic coefficient comparison in original residual coordinates with Euclidean unembedding norm. Input-mode error Gram energy partition by one input slot in covariance eigenbasis; cross-band interactions counted across their two slots. No affine/function/causal equivalence claim. Low-variance concentration is diagnostic, not proof it causes native behavior regression.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
if __name__=='__main__':main()
