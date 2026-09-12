#!/usr/bin/env python3
# BQGATE:384bodyforwards;96prefixes<=29tokens;180seconds;no fitting.
"""pred_a nativewritercontrast>=.2,>=10/12positiveeachorder; scalar<=1e-5,replay<=1e-4.
pred_b BOTHorders removal/writerdonor>=.5coverage,>=10/12 and>=40/48positive,
unrelatedratio<=.5. pred_c readerdonor effectnorm<=.25writerdonoreffectnorm.
Null: confirmedregionalcomponent is not stronglywriter-specific or is order-dependent.
Price384bodyforwards96rows4arms,180seconds,no fitting.
"""
import os,sys,json,time,signal,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from scalar_joint_key_paths_v1 import paths
STEM='SCALAR_EVEN_ROLE_V1';FOLDER=P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/(STEM+'_ROWS.json')).read_text())['rows'];assert len(rows)==96 and max(len(r['ids']) for r in rows)<=29
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('384bodyforwards96rows4arms independentwriter/readerinterchange');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 spec=importlib.util.spec_from_file_location('even_runtime',FOLDER/'execute.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();program={k:v.cuda() for k,v in torch.load(FOLDER/'program.pt',weights_only=True,map_location='cpu').items()};old={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()};bands=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'].cuda();context={};native={};checks=[];count=0
 def hook(index,args,output):
  i=context['row'];arm=context['arm'];s=runtime.scalar(args[0],context['tokens'],program,index)
  if arm==0:
   ref=paths(args[0],context['tokens'],old,index,bands[index])[...,[0,4,5,6,7,8,9],1 if index==0 else 0].sum(-1);assert float(ref.norm())>1e-8;checks.append(float((s-ref).norm()/ref.norm()));native[i,index]=s;return output
  delta=s if arm==1 else s-native[rows[i]['writer_donor' if arm==2 else 'reader_donor'],index]
  return output[0]-(delta[...,None]*program['writers'][index]).to(output[0].dtype),output[1]
 handles=[model.transformer.h[l].attn.register_forward_hook(lambda m,a,o,i=i:hook(i,a,o)) for i,l in enumerate((8,9))];values=torch.zeros(96,4,2,dtype=torch.float64)
 def forward(i,arm):
  nonlocal count
  row=rows[i];tokens=torch.tensor([row['ids']],device='cuda');context.update(tokens=tokens,row=i,arm=arm);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
  for block in model.transformer.h:x,v1=block(x,v1,x0)
  count+=1;logits=(30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30))[0];values[i,arm,0]=(logits[row['uk_id']]-logits[row['us_id']]).cpu();values[i,arm,1]=(logits[row['control_ids'][0]]-logits[row['control_ids'][1]]).cpu()
 try:
  for start in range(0,96,4):
   native.clear()
   for i in range(start,start+4):forward(i,0)
   for i in range(start,start+4):
    for arm in range(1,4):forward(i,arm)
 finally:
  for h in handles:h.remove()
 assert count==384
 priorrows=json.loads((P/'SCALAR_JOINT_KEY_CONFIRMATION_V1_ROWS.json').read_text())['regional'];prior=torch.load(P/'SCALAR_JOINT_KEY_GRADES_CONFIRMATION_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['regional'];lookup={tuple(r['ids']):i for i,r in enumerate(priorrows)};newidx=[i for i,r in enumerate(rows) if tuple(r['ids']) in lookup];oldidx=[lookup[tuple(rows[i]['ids'])] for i in newidx];assert len(newidx)==24
 replay=float((values[newidx,0]-prior[oldidx,0]).norm()/prior[oldidx,0].norm());cells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family];m=values[ix,:,0];grid=m.reshape(12,2,2,4);writer=(grid[:,0]-grid[:,1]).mean(1);reader=(grid[:,:,0]-grid[:,:,1]).mean(1);interaction=grid[:,0,0]-grid[:,0,1]-grid[:,1,0]+grid[:,1,1];base=writer[:,0];reduction=base-writer[:,1];effect=m-m[:,:1];sign=torch.tensor([-1 if rows[i]['writer']==0 else 1 for i in ix],dtype=torch.float64);directed=effect[:,2]*sign;coverage=float(reduction.mean()/base.mean());transfer=float(directed.mean()/base.mean());ratios=[float((values[ix,a,1]-values[ix,0,1]).abs().mean()/effect[:,a].abs().mean().clamp_min(1e-30)) for a in (1,2)];wn=float(effect[:,2].norm());rn=float(effect[:,3].norm());relative=rn/max(wn,1e-30)
  cells.append(dict(family=family,native_writer_mean=float(base.mean()),native_writer_positive=int((base>0).sum()),native_reader_mean=float(reader[:,0].mean()),native_reader_writer_mean_ratio=float(reader[:,0].mean()/base.mean()),native_capability=float(base.mean())>=.2 and int((base>0).sum())>=10,removal_coverage=coverage,positive_removal=int((reduction>0).sum()),writer_donor_transfer=transfer,positive_writer_donor=int((directed>0).sum()),unrelated_ratios=ratios,writer_donor_norm=wn,reader_donor_norm=rn,reader_to_writer_donor_norm=relative,writer_pass=coverage>=.5 and int((reduction>0).sum())>=10 and transfer>=.5 and int((directed>0).sum())>=40 and max(ratios)<=.5,specificity_pass=wn>=1e-5 and relative<=.25,writer_contrast_by_arm=writer.mean(0).tolist(),reader_contrast_by_arm=reader.mean(0).tolist(),interaction_by_arm=interaction.mean(0).tolist()))
 A=max(checks)<=1e-5 and replay<=1e-4 and all(c['native_capability'] for c in cells)
 result={'pred_a':A,'pred_b':A and all(c['writer_pass'] for c in cells),'pred_c':A and all(c['specificity_pass'] for c in cells),'max_scalar_error':max(checks),'native_prior_prompt_replay':replay,'prior_prompt_count':len(newidx),'cells':cells,'body_forwards':count,'seconds':time.perf_counter()-tic,'scope':'Frozencomponent independentwriter/reader interventions, no alternativecandidate/refit. Generalregional/newline results retained regardless of rolespecificity. Paired prompts partlyreused; notcorpusOOD.','source_shas':binding}
 torch.save(dict(margins=values),art);result['artifact_sha']=digest(art);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
