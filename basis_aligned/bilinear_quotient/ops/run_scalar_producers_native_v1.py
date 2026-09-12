#!/usr/bin/env python3
# BQGATE:8 body forwards;48 sequences18-22tokens;compiled producer checks;180seconds.
"""pred_a compiled individual contribution <=1e-5 relative EACHbatch.
pred_b token first-value lookup vs native first-input readings <=1e-5.
pred_c joint contributions and saved pre-MLP17 replay <=1e-5.
Null: compiled implementation invalid; no extraction claim.
Price8bodyforwards, frozen5.54MBprogram,180seconds,under1MBnewartifact.
"""
import os,sys,json,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest
from compiled_scalar_producers_v1 import head_scalar
from regional_cue_row_check_v1 import validate
STEM='SCALAR_PRODUCERS_NATIVE_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(digest(k)==v for k,v in binding.items())
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];validate(rows);batches=[]
 for family in range(2):
  for length in sorted(set(len(r['ids']) for r in rows)):
   ids=[i for i,r in enumerate(rows) if r['family']==family and len(r['ids'])==length];batches.extend((length,ids[o:o+8]) for o in range(0,len(ids),8))
 assert len(batches)==8 and len(rows)==48
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('8bodyforwards48sequences; frozen scalar producers and fullvocab table nativecheck');return
 out=P/(STEM+'_RESULT.json');art=P/(STEM+'_ARTIFACT.pt');assert not out.exists() and not art.exists();tic=time.perf_counter();signal.alarm(180);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();last=model.transformer.h[17]
 program={k:v.cuda() for k,v in torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu').items()}
 reference=torch.load(P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');merged=torch.load(P/'STRUCTURED_PRODUCER_SHARED_OUTPUT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');bank=torch.load(P/'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');first_readers=bank['source_readers'][[2,17],1152:].cuda()
 contributions=torch.zeros(2,48,22,4,dtype=torch.float64);errors=[];first_errors=[];pre_errors=[];captured={}
 def first_hook(module,args):
  actual=args[0].double()@first_readers.T;expected=program['first_token_values'][captured['tokens']];first_errors.append(float((expected-actual).norm()/actual.norm()))
 handles=[model.transformer.h[0].attn.register_forward_pre_hook(first_hook)]
 def hook(index,module,args,out):
  scalar=head_scalar(args[0],captured['tokens'],program,index)
  contribution=scalar[...,None]*program['output_coefficients'][index]*program['shared_output'];ids=captured['ids'];length=contribution.shape[1]
  ref=reference['head_reads'][[2,17][index],ids,:length].cuda()@merged['projectors'][index].cuda().T
  errors.append(dict(producer=index,error=float((contribution-ref).norm()/ref.norm())))
  contributions[index,ids,:length]=contribution.cpu()
 for index,layer in enumerate((8,9)):handles.append(model.transformer.h[layer].attn.register_forward_hook(lambda module,args,out,index=index:hook(index,module,args,out)))
 try:
  for length,ids in batches:
   tokens=torch.tensor([rows[i]['ids'] for i in ids],device='cuda');captured.update(tokens=tokens,ids=ids);x=F.rms_norm(model.transformer.wte(tokens),(1152,));x0=x;v1=None
   for block in model.transformer.h[:17]:x,v1=block(x,v1,x0)
   residual=last.lambdas[0]*x+last.lambdas[1]*x0;attn,_=last.attn(F.rms_norm(residual,(1152,)),v1);pre=(residual+attn)[:,-1].cpu();old=reference['pre'][ids];pre_errors.append(float((pre-old).norm()/old.norm()))
 finally:
  for h in handles:h.remove()
 ref=torch.einsum('hnti,hji->hntj',reference['head_reads'][[2,17]],merged['projectors']);joint_error=float((contributions.sum(0)-ref.sum(0)).norm()/ref.sum(0).norm())
 torch.save(dict(contributions=contributions),art)
 result={'pred_a':max(r['error'] for r in errors)<=1e-5,'pred_b':max(first_errors)<=1e-5,'pred_c':max(pre_errors+[joint_error])<=1e-5,'individual_replay':errors,'first_lookup_replay_max':max(first_errors),'joint_replay':joint_error,'pre_replay_max':max(pre_errors),'seconds':time.perf_counter()-tic,'artifact_sha':digest(art),'source_shas':binding,'scope':'Native validation of standalone scalar producer executor on frozen fresh contexts. Native layer8/9 normalized inputs retained; first-state input replaced by complete token lookup. No fullmodel closure.'}
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}),flush=True)
if __name__=='__main__':main()
