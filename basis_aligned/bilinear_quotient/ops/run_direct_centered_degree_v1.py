#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_lowdegree pred_c_transfer
"""Centered quartic degrees: pred_a_replay pred_b_lowdegree pred_c_transfer."""
import os,sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
PLAN=dict(empirical_panels=2,rows_per_panel=2048,gaussian_rows=1024,batch=128,native_forwards=0)
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print(json.dumps(PLAN));return
 import torch
 sys.path.insert(0,str(P))
 from centered_quartic import degree_terms,check
 from quartic_cp import directional
 validation=check();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=P/'NATIVE_CENTERED_DEGREE_V1.json';assert not out.exists();start=time.perf_counter();state=torch.load('/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin',weights_only=True);capture=torch.load(P/'NATIVE_QUARTIC_COVARIANCE_V1.pt',weights_only=True)['panels']
 def w(k):return state[k].cuda().float()
 with torch.no_grad():
  _,ru=torch.linalg.qr(w('lm_head.weight'));scale=math.sqrt(json.load(open(P/'NATIVE_QUARTIC_QUERY_V1.json'))['stratified_energy']);teacher=[ru@w('transformer.h.17.mlp.Down.weight')/scale,w('transformer.h.17.mlp.Left.weight'),w('transformer.h.17.mlp.Right.weight'),w('transformer.h.16.mlp.Down.weight')*w('transformer.h.17.lambdas')[0],w('transformer.h.16.mlp.Left.weight'),w('transformer.h.16.mlp.Right.weight')];mu=capture[0]['mean'].cuda();cov=capture[0]['covariance'].cuda().double();ev,V=torch.linalg.eigh((cov+cov.T)/2);L=(V*ev.clamp_min(0).sqrt()).float();gen=torch.Generator(device='cuda');gen.manual_seed(1846);panels=[(f'empirical_{n}',p['rows'].cuda()) for n,p in enumerate(capture)]+[('noncentral_gaussian',torch.randn(1024,1152,device='cuda',generator=gen)@L.T+mu)];rows=[]
  for name,x in panels:
   terms=torch.cat([degree_terms(teacher,z,mu) for z in x.split(128)]).double();target=torch.cat([directional(*teacher,[z]*4) for z in x.split(128)]).double();norm=target.square().sum();cross=torch.einsum('nkv,nlv->kl',terms,terms)/norm;cumulative=terms.cumsum(1);errors=[float((cumulative[:,k]-target).norm()/norm.sqrt()) for k in range(5)];ref=degree_terms([v.double() for v in teacher],x[:32].double(),mu.double());precision=float((terms[:32]-ref).norm()/ref.norm());radius=x.double().square().sum(1);row=dict(panel=name,component_cross_energy=cross.tolist(),component_relative_norms=cross.diag().clamp_min(0).sqrt().tolist(),cumulative_relative_errors=errors,precision_error=precision,radius_squared_mean=float(radius.mean()),radius_squared_std=float(radius.std()));rows.append(row);print(json.dumps(row),flush=True)
  predictions=dict(pred_a_replay=all(r['cumulative_relative_errors'][-1]<1e-4 and r['precision_error']<1e-4 for r in rows),pred_b_lowdegree=rows[1]['cumulative_relative_errors'][2]<.5,pred_c_transfer=abs(rows[0]['cumulative_relative_errors'][2]-rows[1]['cumulative_relative_errors'][2])<.15);out.write_text(json.dumps(dict(plan=PLAN,validation=validation,records=rows,predictions=predictions,seconds=time.perf_counter()-start,scope='Exact native quartic expansion about calibration mean; truncateddegree baselines andnonorthogonalcomponent energies. No compact circuit price or fullmodelclaim.'),indent=2)+'\n')
if __name__=='__main__':main()
