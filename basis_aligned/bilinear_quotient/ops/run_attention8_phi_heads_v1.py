#!/usr/bin/env python3
# BQGATE:96bodyforwards;96prefixes<=26tokens;120seconds;native headreader cache.
"""pred_a native margins<=1e-4 andsumheadH<=1e-5relative.
pred_b summed head donorfield reproduces physicalH-only<=1e-4.
pred_c head8.2 nearpostfield predicts fullH-only<=.2relative.
Null: knownhead8.2 alone doesnot supply this partnerinput signal.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest
STEM='ATTENTION8_PHI_HEADS_V1'
@torch.no_grad()
def main():
 files=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in files.items());old=json.loads((P/'SCALAR_NEW_ENDPOINTS_V1_ROWS.json').read_text())['rows'][:24];fresh=json.loads((P/'MLP8_VALUE_FRESH_V1_ROWS.json').read_text())['rows'];rows=old+[dict(r,donor_id=r['donor_id']+24) for r in fresh];validate(rows);assert len(rows)==96
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('96nativeforwards;9head fourreading cache andH-onlywrite replay');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();signal.alarm(120);tic=time.perf_counter();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();fold=torch.load(P/'ATTENTION8_PHI_READER_FOLD_V1_PROGRAM.pt',weights_only=True);C=fold['head_output_readers'].cuda();parent=torch.load(P/'MLP7_PHI_PARENTS_V1_ARTIFACT.pt',weights_only=True);prov=torch.load(P/'PHI4_PROVENANCE_V1_ARTIFACT.pt',weights_only=True);nmax=parent['phi_native'].shape[1];headreads=torch.zeros(96,nmax,9,4,dtype=torch.float64);reg=torch.zeros(96,2,dtype=torch.float64);ctx={};count=0
 def capture(module,args):
  i=ctx['i'];z=args[0].double().reshape(1,-1,9,128);read=torch.einsum('bthk,hik->bthi',z,C);headreads[i,:z.shape[1]]=read[0].cpu()
 handle=model.transformer.h[8].attn.c_proj.register_forward_pre_hook(capture)
 try:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']],device='cuda');ctx['i']=i;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;v1=None
   for block in model.transformer.h:x,v1=block(x,v1,x0)
   logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];count+=1;reg[i,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();reg[i,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 finally:handle.remove()
 assert count==96
 rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30));marginerr=rel(reg,parent['regional']);headerr=rel(headreads.sum(2),parent['parents'][:,:,2]);gen=torch.load(P/'SCALAR_VALUE_GENERATOR_MLP8_V1_PROGRAM.pt',weights_only=True);eig=gen['eigenvalues'][:4];gain=float(gen['lambdas'][0]);fields=torch.zeros(96,nmax,9,dtype=torch.float64)
 for i,row in enumerate(rows):
  n=len(row['ids']);j=row['donor_id'];c=int(prov['cue_positions'][i]);dh=headreads[j,:n]-headreads[i,:n];q=parent['parents'][i,:n,1];value=2*(dh*q[:,None,:]*eig).sum(-1)/parent['rho8_squared'][i,:n,None];value[:c+1]=0;dv=value*gain/prov['rho9'][i,:n,None];fields[i,:n]=prov['gamma'][i,:n,:n]@dv
 target=torch.load(P/'MLP7_QPATH_PORT_DONATION_V1_ARTIFACT.pt',weights_only=True)['donor_fields'][:,3];field_error=rel(fields.sum(-1),target);near=rel(fields[48:72,:,2],target[48:72]);A=marginerr<=1e-4 and headerr<=1e-5;B=A and field_error<=1e-4;Cpass=B and near<=.2
 result={'pred_a':A,'pred_b':B,'pred_c':Cpass,'margin_replay_error':marginerr,'head_sum_error':headerr,'donor_field_replay_error':field_error,'head8_2_nearpost_field_error':near,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Nativehead output-readings and exactconditionalH-onlydonor field partition. No individualhead physicalintervention or globalmoduleclaim.'}
 torch.save(dict(head_readings=headreads,head_donor_fields=fields,regional=reg),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
