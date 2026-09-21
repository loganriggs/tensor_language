#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument pred_b_allocation pred_c_context
"""Matched-product allocation for centered original operator plus exact first-order terms.
pred_a_instrument mean expansion/helper/replay controls <1e-8.
pred_b_allocation width512rank1 context error<width256rank2 at same512products bothdomains.
pred_c_context width512rank2 context error<prior512 program bothdomains.
512 SVDs, 48 captures, no fitting on native diagnostics. Includes full linear-map costs.
"""
import os,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal/direct_tensor_match'
def main():
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):
  print(json.dumps(dict(forwards=48,svds=512,dimension=1152,products=[512,1024],fit_held=False)));return
 import torch
 sys.path.insert(0,str(P));from weighted_bilinear_svd import roots
 from midpoint_program import product_source_delta
 from circuit_fast_screen_producer import Bilin18TorchBackend
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;start=time.perf_counter()
 out=P/'MIDPOINT_CENTERED_ALLOCATION_V1.json';assert not out.exists()
 model=Bilin18TorchBackend.load('cuda').model.float();_,ru=torch.linalg.qr(model.lm_head.weight.double(),mode='reduced');block=model.transformer.h[17];L=block.mlp.Left.weight.double();R=block.mlp.Right.weight.double();C=ru@block.mlp.Down.weight.double()
 rows=torch.load(P/'MIDPOINT_CALIBRATION_ROWS_V1.pt',weights_only=True);n=rows['n'].flatten(0,1).cuda().double();m=rows['m'].flatten(0,1).cuda().double();nb=n.mean(0);mb=m.mean(0);nc=n-nb;mc=m-mb
 Sn,In=roots(nc.T@nc/len(n));Sm,Im=roots(mc.T@mc/len(m))
 Jn=(L.T*(mb@R.T))@C.T+(R.T*(mb@L.T))@C.T;Jm=(L.T*(nb@R.T))@C.T+(R.T*(nb@L.T))@C.T;fbar=((nb@L.T)*(mb@R.T)+(nb@R.T)*(mb@L.T))@C.T
 truth=lambda a,b:((a@L.T)*(b@R.T)+(a@R.T)*(b@L.T))@C.T
 checks=[float((truth(n[:32],m[:32])-(truth(nc[:32],mc[:32])+n[:32]@Jn+m[:32]@Jm-fbar)).norm()/truth(n[:32],m[:32]).norm())]
 bases=torch.load(P/'MIDPOINT_CENTERED_OUTPUT_BASES_V1.pt',weights_only=True)['centered512'];q=bases['span_readers'].cuda().double();writers=bases['span_writers'].cuda().double();channel=q.T@C
 Af=[];Bf=[];spectra=[]
 for j in range(512):
  K=(L.T*channel[j])@R+(R.T*channel[j])@L;Z=Sn@K@Sm;U,s,Vh=torch.linalg.svd(Z,full_matrices=False)
  Af.append(((In@U[:,:4])*s[:4].sqrt()).cpu());Bf.append(((Im@Vh[:4].T)*s[:4].sqrt()).cpu());spectra.append(dict(direction=j,total_energy=float(s.square().sum()),tail1=float(s[1:].square().sum()),tail2=float(s[2:].square().sum()),tail4=float(s[4:].square().sum())))
  if (j+1)%32==0:print('decomposed',j+1,'seconds',time.perf_counter()-start,flush=True)
 programs={'prior512':torch.load(P/'MIDPOINT_GRAPH_PRODUCT_PRUNE_V1.pt',weights_only=True)['products512']};prices=[]
 for width,rank in [(256,2),(512,1),(256,4),(512,2)]:
  A=torch.cat([a[:,:rank] for a in Af[:width]],1);B=torch.cat([a[:,:rank] for a in Bf[:width]],1);count=width*rank
  e=dict(A=A,B=B,base_left_mean=nb.cpu()@A,base_right_mean=mb.cpu()@B,group_writers=writers[:,:width].cpu(),correction_left=torch.zeros(count,0,dtype=torch.float64),correction_writers=torch.zeros(1152,0,dtype=torch.float64),product_mean=torch.zeros(count,dtype=torch.float64),full_mean=-fbar.cpu(),linear_n=Jn.cpu(),linear_m=Jm.cpu())
  programs[f'w{width}r{rank}']=e;prices.append(dict(width=width,rank=rank,products=count,weight_coefficients=2*1152*count+1152*width+2*1152**2,mean_state=2*count+1152))
  a=n[:16].cpu();b=m[:16].cpu();donor=b.roll(3,0)
  def evaluate(x,z):
   phi=(x@A-e['base_left_mean'])*(z@B-e['base_right_mean'])
   return phi.reshape(len(x),width,rank).sum(-1)@e['group_writers'].T+x@e['linear_n']+z@e['linear_m']+e['full_mean']
  direct=evaluate(a,donor)-evaluate(a,b);fast=product_source_delta(e,a,donor-b);checks.append(float((direct-fast).norm()/direct.norm()))
 assert max(checks)<1e-8
 artifact=P/'MIDPOINT_CENTERED_ALLOCATION_GRAPHS_V1.pt';assert not artifact.exists();torch.save(programs,artifact)
 del model,L,R,C,n,m,nc,mc,Sn,Sm,In,Im,U,s,Vh,Z,K,Jn,Jm,writers;torch.cuda.empty_cache()
 import run_direct_midpoint_full_replace_v1 as native
 native.EXTRA_PRODUCT_FILE=artifact.name;native.SOURCE_CONTEXT_FAMILIES=True;native.OUTPUT_NAME='MIDPOINT_CENTERED_ALLOCATION_NATIVE_RAW_V1.json';native.main()
 raw=json.loads((P/native.OUTPUT_NAME).read_text());ss=raw['summary'];value=lambda d,k:ss[d]['product_'+k]['context_only']['centered_effect_relative_error']
 pred=dict(pred_a_instrument=max(checks)<1e-8 and raw['predictions']['pred_a_instrument'],pred_b_allocation=all(value(d,'w512r1')<value(d,'w256r2') for d in ss),pred_c_context=all(value(d,'w512r2')<value(d,'prior512') for d in ss))
 out.write_text(json.dumps(dict(predictions=pred,checks=checks,prices=prices,spectra=spectra,summary=ss,seconds=time.perf_counter()-start,scope='Original centered operator per-output weighted SVD, exact original mean/single-input terms explicitly stored and priced. Centered256/512 from calibration-only weight contractions. Shared output directions; no input sharing beyond learned product factors. Matched variable-product counts not matched total storage. Reused diagnostic panels; no semantic/OOD adoption.'),indent=2)+'\n');print(json.dumps(pred))
if __name__=='__main__':main()
