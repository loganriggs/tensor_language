#!/usr/bin/env python3
# BQGATE:384bodyforwards;96prefixes<=29tokens;180seconds;fresh conditional path test.
"""pred_a old native/full/first marginreplay<=1e-4 andtokenvaluereplay<=1e-5.
pred_b eachfresh nativecontrastmean>.05,positive>=10/12; firstvalue effects
expected sign>=20/24: near families negative,distant positive.
pred_c eachfresh firstvalueeffecterror<=.4fullhead8.2 andhalf-strength
linearityerror<=.1. Null: syntax changes the path sign or approximation.
Price384forwards180sec; nativecontext generated, no activationfit.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from regional_even_routing_v1 import routing
from sparse_path_stability_atlas_v1 import digest
STEM='FIRST_TOKEN_PATH_FRESH_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1';TOKEN=P/'extracted_circuits/first_token_value_path_v1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];validate(rows);assert len(rows)==96
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('384nativeforwards:96nativecache+3interventions each;72fresh prefixes');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();fp={k:v.cuda() for k,v in torch.load(P/'MLP7_PHI_READERS_FOLD_V1_PROGRAM.pt',weights_only=True).items()};fold=torch.load(P/'ATTENTION8_PHI_READER_FOLD_V1_PROGRAM.pt',weights_only=True);C=fold['head_output_readers'][2].cuda();program={k:v.cuda() for k,v in torch.load(TOKEN/'program.pt',weights_only=True).items()};p={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True).items()};spec=importlib.util.spec_from_file_location('token_path_executor',TOKEN/'execute.py');exe=importlib.util.module_from_spec(spec);spec.loader.exec_module(exe);nmax=max(len(r['ids']) for r in rows);device='cuda';q=torch.zeros(96,nmax,4,dtype=torch.float64,device=device);hh=torch.zeros_like(q);fv=torch.zeros_like(q);rho8=torch.zeros(96,nmax,dtype=torch.float64,device=device);rho9=torch.zeros_like(rho8);g8=torch.zeros(96,nmax,nmax,dtype=torch.float64,device=device);g9=torch.zeros_like(g8);reg=torch.zeros(96,4,2,dtype=torch.float64);ctx={'capture':True,'arm':0};token_errors=[];fields=torch.zeros(96,2,nmax,dtype=torch.float64,device=device);attn8=model.transformer.h[8].attn;original=attn8.squared_attention
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 def mlp7(module,args,output):
  if ctx['capture']:
   x=args[0].double();n=x.shape[1];q[ctx['i'],:n]=(((x@module.Left.weight.double().T)*(x@module.Right.weight.double().T))@fp['product_coefficients'].T*fp['lambda8'][0])[0]
 def pre8(module,args):
  if ctx['capture']:ctx['raw8']=module.lambdas[0]*args[0]+module.lambdas[1]*args[2]
 def out8(module,args,output):
  if ctx['capture']:
   z=ctx['raw8']+output[0];rho8[ctx['i'],:z.shape[1]]=z.square().mean(-1).double()[0]+torch.finfo(torch.float32).eps
 def prevalue8(module,args):
  if ctx['capture']:
   x,v1=args;n=x.shape[1];fv[ctx['i'],:n]=((module.lamb*v1.view(1,n,9,128))[:,:,2].double()@C.T)[0]
 def headread(module,args):
  if ctx['capture']:
   n=args[0].shape[1];hh[ctx['i'],:n]=(args[0].view(1,n,9,128)[:,:,2].double()@C.T)[0]
 def joint(q1,k1,v,q2,k2):
  if ctx['capture']:
   n=q1.shape[1];a=torch.einsum('btd,bsd->bts',q1[:,:,2],k1[:,:,2])/128;b=torch.einsum('btd,bsd->bts',q2[:,:,2],k2[:,:,2])/128;g8[ctx['i'],:n,:n]=(a*b).masked_fill(~torch.ones(n,n,device=device,dtype=torch.bool).tril(),0)[0].double()
  return original(q1,k1,v,q2,k2)
 def pre9(module,args):
  if ctx['capture']:
   z=module.lambdas[0]*args[0]+module.lambdas[1]*args[2];rho9[ctx['i'],:z.shape[1]]=(z.square().mean(-1)+torch.finfo(torch.float32).eps).sqrt().double()[0]
 def attpre9(module,args):
  if ctx['capture']:
   n=args[0].shape[1];g9[ctx['i'],:n,:n]=routing(args[0],p,1)[0]
 def write9(module,args,output):
  if ctx['capture'] or ctx['arm']==0:return output
  arm=ctx['arm'];n=args[0].shape[1];value=fields[ctx['i'],0 if arm==1 else 1,:n]*(.5 if arm==3 else 1.);return output[0]+(value[None,:,None]*p['writers'][1]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[7].mlp.register_forward_hook(mlp7),model.transformer.h[8].register_forward_pre_hook(pre8),attn8.register_forward_hook(out8),attn8.register_forward_pre_hook(prevalue8),attn8.c_proj.register_forward_pre_hook(headread),model.transformer.h[9].register_forward_pre_hook(pre9),model.transformer.h[9].attn.register_forward_pre_hook(attpre9),model.transformer.h[9].attn.register_forward_hook(write9)];attn8.squared_attention=joint;count=0
 def forward(i,arm):
  row=rows[i];ids=torch.tensor([row['ids']],device=device);ctx.update(i=i,arm=arm);x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];reg[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 try:
  for i in range(96):forward(i,0);count+=1
  for i,row in enumerate(rows):
   n=len(row['ids']);j=row['donor_id'];ids=torch.tensor(row['ids'],device=device);did=torch.tensor(rows[j]['ids'],device=device);cue=int(torch.where(ids!=did)[0].item());mask=torch.arange(n,device=device)>cue;token_errors.append(rel(exe.token_readings(ids,model.transformer.wte.weight,program),fv[i,:n]));fields[i,1,:n]=exe.donor_field(ids,did,model.transformer.wte.weight,program,g8[i,:n,:n],q[i,:n],rho8[i,:n],g9[i,:n,:n],rho9[i,:n],mask);dh=hh[j,:n]-hh[i,:n];value=2*(q[i,:n]*program['eigenvalues']*dh).sum(-1)/rho8[i,:n]*mask;fields[i,0,:n]=g9[i,:n,:n]@(program['head9_gain']*value/rho9[i,:n])
  ctx['capture']=False
  for i in range(96):
   for arm in range(1,4):forward(i,arm);count+=1
 finally:
  for h in handles:h.remove()
  attn8.squared_attention=original
 assert count==384
 old=torch.load(P/'ATTENTION8_PHI_VALUE_PORTS_V1_ARTIFACT.pt',weights_only=True)['regional'][48:72][:,[0,1,6]];replay=rel(reg[:24,:3],old);records=[]
 for group in range(4):
  z=reg[group*24:(group+1)*24];native=z[::2,0,0]-z[1::2,0,0];effects=[];arms=[]
  for arm in range(1,4):
   d=z[:,arm,0]-z[:,0,0];e=d.clone();e[::2]*=-1;effects.append(e);arms.append(dict(arm=arm,transfer=float(e.mean()/native.mean()),positive=int((e>0).sum()),negative=int((e<0).sum()),unrelated=float((z[:,arm,1]-z[:,0,1]).abs().mean()/d.abs().mean().clamp_min(1e-30))))
  records.append(dict(group=group,native_mean_contrast=float(native.mean()),native_positive_pairs=int((native>0).sum()),arms=arms,first_relative_to_full=rel(effects[1],effects[0]),half_linearity_error=rel(2*effects[2],effects[1])))
 A=replay<=1e-4 and max(token_errors)<=1e-5;B=A and all(r['native_mean_contrast']>.05 and r['native_positive_pairs']>=10 and r['arms'][1]['negative' if r['group']<3 else 'positive']>=20 for r in records[1:]);C=A and all(r['first_relative_to_full']<=.4 and r['half_linearity_error']<=.1 for r in records[1:]);result={'pred_a':A,'pred_b':B,'pred_c':C,'old_anchor_replay_error':replay,'max_token_generator_error':max(token_errors),'records':records,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'72untouched fullprefixes plus24oldanchors. Fixedweight token input generator andnativecontext ports; physical full/first/half path edits. Syntax generalization test, not corpusOOD/autonomousextraction.'}
 torch.save(dict(regional=reg,fields=fields.cpu(),q7=q.cpu(),head8_readings=hh.cpu(),rho8_squared=rho8.cpu(),rho9=rho9.cpu(),gamma8=g8.cpu(),gamma9=g9.cpu()),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
