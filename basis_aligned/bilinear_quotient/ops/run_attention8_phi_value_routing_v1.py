#!/usr/bin/env python3
# BQGATE:96bodyforwards;96prefixes<=26tokens;120seconds;native value/routing cache.
"""pred_a native margin<=1e-4 and head8.2 readings replay<=1e-5relative.
pred_b H-only donorfield replay<=1e-4relative.
pred_c first-layer generatedH donorfield nearquote<=.2relative fullhead8.2.
Null: current values or routing/value interactions are necessary.
Price96nativeforwards120sec, about1MBcache; fullnativebackground required.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='ATTENTION8_PHI_VALUE_ROUTING_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items())
 old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows);assert len(rows)==96
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('96 native forwards; joint routing and current/first value cache; no native output edits');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();tic=time.perf_counter();fold=torch.load(P/'ATTENTION8_PHI_READER_FOLD_V1_PROGRAM.pt',weights_only=True);C=fold['head_output_readers'][2].cuda();parent=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);prior=torch.load(P/'ATTENTION8_PHI_HEADS_V1_ARTIFACT.pt',weights_only=True);nmax=parent['phi_native'].shape[1]
 values=torch.zeros(96,nmax,2,4,dtype=torch.float64);routing=torch.zeros(96,nmax,nmax,dtype=torch.float64);reg=torch.zeros(96,2,dtype=torch.float64);ctx={};attn=model.transformer.h[8].attn;original=attn.squared_attention;assert attn.squared_attn
 def capture_values(module,args):
  i=ctx['i'];x,v1=args;n=x.shape[1];assert v1 is not None
  vc=((1-module.lamb)*module.c_v(x).view(1,n,9,128))[:,:,2].double();vf=(module.lamb*v1.view(1,n,9,128))[:,:,2].double()
  values[i,:n,0]=(vc@C.T)[0].cpu();values[i,:n,1]=(vf@C.T)[0].cpu()
 def capture_routing(q,k,v,q2,k2):
  n=q.shape[1];scores=torch.einsum('btd,bsd->bts',q[:,:,2],k[:,:,2])/128;scores2=torch.einsum('btd,bsd->bts',q2[:,:,2],k2[:,:,2])/128;gamma=scores*scores2;gamma=gamma.masked_fill(~torch.ones(n,n,device=q.device,dtype=torch.bool).tril(),0);routing[ctx['i'],:n,:n]=gamma[0].double().cpu();return original(q,k,v,q2,k2)
 handle=attn.register_forward_pre_hook(capture_values);attn.squared_attention=capture_routing;count=0
 try:
  for i,row in enumerate(rows):
   ctx['i']=i;ids=torch.tensor([row['ids']],device='cuda');x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
   for block in model.transformer.h:x,v1=block(x,v1,x0)
   logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:handle.remove();attn.squared_attention=original
 assert count==96
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));readings=torch.einsum('bts,bskd->btkd',routing,values);marginerr=rel(reg,prior['regional']);readerr=rel(readings.sum(2),prior['head_readings'][:,:,2]);gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);eig=gen['eigenvalues'][:4];gain=float(gen['lambdas'][0]);fields=torch.zeros(96,nmax,2,dtype=torch.float64)
 for i,row in enumerate(rows):
  n=len(row['ids']);j=row['donor_id'];cue=int(prov['cue_positions'][i]);dh=readings[j,:n]-readings[i,:n];q=parent['parents'][i,:n,1];value=2*(dh*q[:,None,:]*eig).sum(-1)/parent['rho8_squared'][i,:n,None];value[:cue+1]=0;fields[i,:n]=prov['gamma'][i,:n,:n]@(value*gain/prov['rho9'][i,:n,None])
 target=prior['head_donor_fields'][:,:,2];ferr=rel(fields.sum(-1),target);errors=[rel(fields[k*24:(k+1)*24,:,1],target[k*24:(k+1)*24]) for k in range(4)];A=marginerr<=1e-4 and readerr<=1e-5;B=A and ferr<=1e-4
 result={'pred_a':A,'pred_b':B,'pred_c':B and errors[2]<=.2};result.update(margin_replay_error=marginerr,head_reading_replay_error=readerr,field_replay_error=ferr,first_generated_sector_field_errors=errors,body_forwards=count,seconds=time.perf_counter()-tic,scope='Native jointQK routing plus current/first sourcevalues; generatedH sector donation fields, not physical sector input swap. Reused96prefixes; native contextual inputs/background.')
 torch.save(dict(source_values=values,routing=routing,sector_readings=readings,sector_fields=fields,regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
