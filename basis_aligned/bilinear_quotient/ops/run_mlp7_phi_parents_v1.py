#!/usr/bin/env python3
# BQGATE:96bodyforwards;96prefixes<=26tokens;120seconds;no fitting.
"""pred_a native margin replay<=1e-4 and foldedQ/native projection<=1e-4.
pred_b reader/phi replay<=1e-5 andsixpathsum<=1e-10.
pred_c mixedpath omission citycuephi error>=.2 each4groups.
Null: independent parent squares suffice for these city value contrasts.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='MLP7_PHI_PARENTS_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+fresh;validate(rows);assert len(rows)==96
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('96nativeforwards;B/Q7/H8 fourreader andsixphi paths');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();program={k:v.cuda() for k,v in torch.load(P/'MLP7_PHI_READERS_FOLD_V1_PROGRAM.pt',weights_only=True).items()};u=program['readers'];eig=program['eigenvalues'];lam=program['lambda8'];ctx={};nmax=max(len(r['ids']) for r in rows);parents=torch.zeros(96,nmax,3,4,dtype=torch.float64);norm=torch.zeros(96,nmax,dtype=torch.float64);paths=torch.zeros(96,nmax,6,dtype=torch.float64);reg=torch.zeros(96,2,dtype=torch.float64);phis=torch.zeros(96,nmax,dtype=torch.float64);checks=[];count=0
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 def pre7(module,args):ctx['raw7']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def att7(module,args,output):ctx['z7']=ctx['raw7']+output[0]
 def mlp7(module,args,output):
  x=args[0].double();q=((x@module.Left.weight.double().T)*(x@module.Right.weight.double().T))@program['product_coefficients'].T;ctx['Q']=q*lam[0];ctx['qerr']=rel(q,(output-module.Down_bias).double()@u)
 def pre8(module,args):
  ctx['raw8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['B']=lam[0]*(ctx['z7'].double()@u)+lam[1]*(args[2].double()@u)+lam[0]*program['bias_reads']
 def att8(module,args,output):ctx['H']=output[0].double()@u;ctx['z8']=ctx['raw8']+output[0]
 def mlp8(module,args):
  i=ctx['i'];n=args[0].shape[1];b,q,h=ctx['B'],ctx['Q'],ctx['H'];rho=ctx['z8'].square().mean(-1).double()+torch.finfo(torch.float32).eps;target=((args[0].double()@u).square()*eig).sum(-1);pred=((b+q+h).square()*eig).sum(-1)/rho
  pp=torch.stack([(b.square()*eig).sum(-1),(q.square()*eig).sum(-1),(h.square()*eig).sum(-1),2*(b*q*eig).sum(-1),2*(b*h*eig).sum(-1),2*(q*h*eig).sum(-1)],-1)/rho[...,None]
  checks.append(dict(q=ctx['qerr'],read=rel(b+q+h,ctx['z8'].double()@u),phi=rel(pred,target),six=rel(pp.sum(-1),pred)));parents[i,:n]=torch.stack([b,q,h],-2)[0].cpu();norm[i,:n]=rho[0].cpu();paths[i,:n]=pp[0].cpu();phis[i,:n]=target[0].cpu()
 handles=[model.transformer.h[7].register_forward_pre_hook(pre7),model.transformer.h[7].attn.register_forward_hook(att7),model.transformer.h[7].mlp.register_forward_hook(mlp7),model.transformer.h[8].register_forward_pre_hook(pre8),model.transformer.h[8].attn.register_forward_hook(att8),model.transformer.h[8].mlp.register_forward_pre_hook(mlp8)]
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
   for block in model.transformer.h:x,v1=block(x,v1,x0)
   logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:
  for h in handles:h.remove()
 assert count==96
 ref=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);marginerr=rel(reg,ref['regional'][:,0]);maxchecks={k:max(c[k] for c in checks) for k in checks[0]};records=[]
 for group in range(4):
  ix=torch.arange(group*24,(group+1)*24);city=ref['cue_positions'][ix];truth=phis[ix,city];independent=paths[ix,city,:3].sum(-1);y=truth[::2]-truth[1::2];yh=independent[::2]-independent[1::2];records.append(dict(group=group,independent_square_citycue_error=rel(yh,y),native_citycue_norm=float(y.norm())))
 A=marginerr<=1e-4 and maxchecks['q']<=1e-4;B=A and max(maxchecks['read'],maxchecks['phi'])<=1e-5 and maxchecks['six']<=1e-10;C=B and all(r['independent_square_citycue_error']>=.2 for r in records)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'margin_replay_error':marginerr,'max_checks':maxchecks,'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Exact native reader-parent and sixpath computation; mixedpath omission oncity contrasts, not newcausal intervention or autonomous generation.'}
 torch.save(dict(parents=parents,rho8_squared=norm,phi_paths=paths,phi_native=phis,regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
