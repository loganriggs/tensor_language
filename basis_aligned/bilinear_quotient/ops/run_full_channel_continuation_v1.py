#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_replay pred_b_observer pred_c_mode
"""Frozen continuation observer inside full-layer compression, zero native forwards.
Native scalar-matrix replay <1e-10; compressed observer covariance error <.10;
leading weighted rank-one outer-product cosine >.99. Null: natural accuracy
conceals this behaviorally grounded mixed interaction. No fitting or adoption.
Price: existing 3686 products, 14,067,072 coefficients including affine branch;
observer is diagnostic, not an extra deployed branch. Isotropic error also shown.
"""
import os,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=0,fit=False,arms=['original_retained','half_refit','full_refit'])));return
 import torch
 torch.set_num_threads(2);torch.set_grad_enabled(False);start=time.monotonic()
 out=P/'FULL_CHANNEL_CONTINUATION_V1.json';assert not out.exists()
 checkpoint='/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin'
 state=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
 L=state['transformer.h.17.mlp.Left.weight'].cuda().double();R=state['transformer.h.17.mlp.Right.weight'].cuda().double();D=state['transformer.h.17.mlp.Down.weight'].cuda().double()
 e={k:v.cuda().double() for k,v in torch.load(P/'MIDPOINT_CONTINUATION_NATIVE_OPERATOR_V1.pt',weights_only=True).items()}
 p={k:v.cuda().double() for k,v in torch.load(P/'FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt',weights_only=True).items()}
 scalar_writer=e['q']@e['R_U'];channel=scalar_writer@D
 native=L.T@(channel[:,None]*R)+R.T@(channel[:,None]*L)
 replay=float((native-e['native_matrix']).norm()/native.norm())
 ids=torch.tensor(next(r['selected_channels'] for r in json.loads((P/'FULL_CHANNEL_DELETION_V1.json').read_text())['records'] if r['geometry']=='activation_covariance' and r['policy']=='conditional' and r['width']==3686),device='cuda')
 a,b=p['a'],p['b'];scales=(L[ids].norm(dim=1)/a.norm(dim=1))*(R[ids].norm(dim=1)/b.norm(dim=1));w0=D[:,ids]*scales
 rows=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);inputs=[];roots=[]
 for key,mean in [('n','mean_n'),('m','mean_m')]:
  x=rows[key].flatten(0,1).cuda().double()-e[mean];inputs.append(x);ev,V=torch.linalg.eigh(x.T@x/len(x));cut=ev.clamp_min(0)*(ev>1e-10*ev.max());roots.append((V*cut.sqrt())@V.T)
 N,M=roots;n,m=inputs;weighted=N@native@M;u,s,vh=torch.linalg.svd(weighted,full_matrices=False);mode=s[0]*torch.outer(u[:,0],vh[0]);truth=((n@native)*m).sum(1);records=[];controls=[]
 for t in (0.,.5,1.):
  writer=(1-t)*w0+t*p['writer'];c=scalar_writer@writer;matrix=a.T@(c[:,None]*b)+b.T@(c[:,None]*a);w=N@matrix@M;us,ss,vs=torch.linalg.svd(w,full_matrices=False);student_mode=ss[0]*torch.outer(us[:,0],vs[0]);scalar=((n@matrix)*m).sum(1)
  direct=((n[:17]@a.T)*(m[:17]@b.T)+(n[:17]@b.T)*(m[:17]@a.T))@c;controls.append(float((direct-scalar[:17]).norm()/direct.norm()))
  records.append(dict(refit_fraction=t,isotropic_observer_error=float((matrix-native).norm()/native.norm()),covariance_observer_error=float((w-weighted).norm()/weighted.norm()),paired_calibration_scalar_error=float((scalar-truth).norm()/truth.norm()),leading_mode_cosine=float((student_mode*mode).sum()/(student_mode.norm()*mode.norm())),leading_mode_relative_error=float((student_mode-mode).norm()/mode.norm()),leading_singular_value_ratio=float(ss[0]/s[0]),leading_energy_fraction=float(ss[0].square()/ss.square().sum())))
 final=records[-1];pred=dict(pred_a_replay=max([replay]+controls)<1e-10,pred_b_observer=final['covariance_observer_error']<.1,pred_c_mode=final['leading_mode_cosine']>.99)
 result=dict(records=records,predictions=pred,native_matrix_replay=replay,maximum_factor_replay=max(controls),native_leading_energy_fraction=float(s[0].square()/s.square().sum()),seconds=time.monotonic()-start,scope='Same fixed original QR-frame scalar observer and centered midpoint coordinates as prior continuation study. Compressed mixed Hessian is formed from its own factors; affine correction has zero mixed Hessian. Covariance product metric and paired historical calibration are not fresh functional or causal validation. Mode cosine is sign-invariant to paired SVD signs, not proof of monosemantic identity.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
