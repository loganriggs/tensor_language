#!/usr/bin/env python3
# BQGATE:256bodyforwards;64prefixes<=248tokens;180seconds;freshFineWebpreservation.
"""pred_a oldallarms CE/marginreplay<=1e-4relative,tokenread<=1e-5.
pred_b freshallsourceKO meanabsCE<=.02,maxabsCE<=.1eachfamily.
pred_c freshnativepositive>=14/16,meanmargin>0,headzero meanabsCE>=.005
and>=10xcandidatemeanabsCEeachfamily. Null: freshcontrolsrevealdamageorweakfixture.
Price256forwards180sec;32oldanchors+32newprefixes. No corpusOODclaim.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_even_routing_v1 import routing
from sparse_path_stability_atlas_v1 import digest
STEM='FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1';TOKEN=P/'extracted_circuits/first_token_value_path_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());rows=json.loads((P/'FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json').read_text())['rows'];assert len(rows)==64
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('256forwards64FineWebrows;native,allsourceKO,halfKO,head8.2zero');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();fp={k:v.cuda() for k,v in torch.load(P/'MLP7_PHI_READERS_FOLD_V1_PROGRAM.pt',weights_only=True).items()};fold=torch.load(P/'ATTENTION8_PHI_READER_FOLD_V1_PROGRAM.pt',weights_only=True);C=fold['head_output_readers'][2].cuda();pr={k:v.cuda() for k,v in torch.load(TOKEN/'program.pt',weights_only=True).items()};p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};s=importlib.util.spec_from_file_location('tok',TOKEN/'execute.py');exe=importlib.util.module_from_spec(s);s.loader.exec_module(exe);ctx={};fields={};ce=torch.zeros(64,4,dtype=torch.float64);margins=torch.zeros_like(ce);errors=[];att8=model.transformer.h[8].attn;original=att8.squared_attention
 def mlp7(module,args,output):
  if ctx['arm']==0:
   x=args[0].double();ctx['Q']=((x@module.Left.weight.double().T)*(x@module.Right.weight.double().T))@fp['product_coefficients'].T*fp['lambda8'][0]
 def pre8(module,args):
  if ctx['arm']==0:ctx['raw8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def prevalue8(module,args):
  if ctx['arm']==0:
   x,v1=args;n=x.shape[1];actual=((module.lamb*v1.view(1,n,9,128))[:,:,2].double()@C.T)[0];ctx['F']=exe.token_readings(ctx['ids'][0],model.transformer.wte.weight,pr);errors.append(float((ctx['F']-actual).norm()/actual.norm()))
 def joint(q,k,v,q2,k2):
  if ctx['arm']==0:
   n=q.shape[1];a=torch.einsum('btd,bsd->bts',q[:,:,2],k[:,:,2])/128;b=torch.einsum('btd,bsd->bts',q2[:,:,2],k2[:,:,2])/128;g=(a*b).masked_fill(~torch.ones(n,n,device=q.device,dtype=torch.bool).tril(),0);ctx['Hfirst']=g.double()@ctx['F']
  return original(q,k,v,q2,k2)
 def preproj(module,args):
  if ctx['arm']==3:
   n=args[0].shape[1];ctx['wholehead']=F.linear(args[0].reshape(1,n,9,128)[:,:,2],module.weight[:,256:384])
 def out8(module,args,output):
  if ctx['arm']==0:
   z=ctx['raw8']+output[0];ctx['R8']=z.square().mean(-1).double()+torch.finfo(torch.float32).eps
  if ctx['arm']==3:return output[0]-ctx['wholehead'],output[1]
  return output
 def pre9(module,args):
  if ctx['arm']==0:
   z=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];ctx['rho9']=(z.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()
 def write9(module,args,output):
  arm=ctx['arm'];i=ctx['i']
  if arm==0:
   value=2*pr['head9_gain']*(ctx['Q']*pr['eigenvalues']*ctx['Hfirst']).sum(-1)/(ctx['R8']*ctx['rho9']);fields[i]=(routing(args[0],p,1)@value[...,None])[...,0];return output
  if arm in (1,2):return output[0]-((1. if arm==1 else .5)*fields[i][...,None]*p['writers'][1]).to(output[0].dtype),output[1]
  return output
 handles=[model.transformer.h[7].mlp.register_forward_hook(mlp7),model.transformer.h[8].register_forward_pre_hook(pre8),att8.register_forward_pre_hook(prevalue8),att8.c_proj.register_forward_pre_hook(preproj),att8.register_forward_hook(out8),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_hook(write9)];att8.squared_attention=joint;count=0
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx.update(i=i,ids=ids)
   for arm in range(4):
    ctx['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
    for block in model.transformer.h:x,v1=block(x,v1,x0)
    logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;ce[i,arm]=-logits.log_softmax(-1)[198].cpu();margins[i,arm]=(logits[198]-logits[11]).cpu()
 finally:
  for h in handles:h.remove()
  att8.squared_attention=original
 assert count==256
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));ref=torch.load(P/'FIRST_TOKEN_REMOVAL_NEWLINE_V1_ARTIFACT.pt',weights_only=True);replay=max(rel(ce[:32],ref['ce']),rel(margins[:32],ref['margins']));records=[]
 for k in range(4):
  z=ce[k*16:(k+1)*16];ma=margins[k*16:(k+1)*16];delta=z[:,1]-z[:,0];records.append(dict(family=k,native_mean_margin=float(ma[:,0].mean()),native_positive=int((ma[:,0]>0).sum()),mean_CE_change=float(delta.mean()),meanabs_CE_change=float(delta.abs().mean()),maxabs_CE_change=float(delta.abs().max()),wholehead_meanabs_CE_change=float((z[:,3]-z[:,0]).abs().mean()),half_CE_linearity_error=rel(2*(z[:,2]-z[:,0]),delta)))
 A=replay<=1e-4 and max(errors)<=1e-5;B=A and all(r['meanabs_CE_change']<=.02 and r['maxabs_CE_change']<=.1 for r in records[2:]);C=A and all(r['native_positive']>=14 and r['native_mean_margin']>0 and r['wholehead_meanabs_CE_change']>=.005 and r['wholehead_meanabs_CE_change']>=10*r['meanabs_CE_change'] for r in records[2:]);result={'pred_a':A,'pred_b':B,'pred_c':C,'native_positive_control_replay_error':replay,'max_token_reader_error':max(errors),'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Frozenallsource interaction on32old+32newFineWebnewlineprefixes; cacheindicesdisjoint,documentindependence notestablished. No modelscorefilter. Prospectivecontrolbar; oldCfailureunchanged. NotcorpusOOD orautonomousextraction.'}
 torch.save(dict(ce=ce,margins=margins),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
