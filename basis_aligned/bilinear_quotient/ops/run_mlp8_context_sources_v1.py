#!/usr/bin/env python3
# BQGATE:40bodyforwards;40prefixes;300seconds;no fitting.
"""pred_a anchor<=1e-5/contextsum<=1e-6; pred_b CPU ordered closure<=1e-10;
pred_c CPU native delta<=1e-4.40native forwards; no causal omission claim.
"""
from pathlib import Path
import hashlib,json,os,sys,signal,time
import torch
ROOT=Path(__file__).resolve().parents[3];P=ROOT/'basis_aligned/polynomial_causal';sys.path[:0]=[str(Path(__file__).parent),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import measure
STEM='MLP8_CONTEXT_SOURCES_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'TYPED_FACE_NATIVE8_FRESH_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows);assert len(groups)==40
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('40bodyforwards;40prefixes;three native context sources');return
 out=P/(STEM+'_RESULT.json');assert not out.exists();start=time.perf_counter();signal.alarm(300);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 from fastload import load_model_fast
 model=load_model_fast().cuda().eval();sources=[];state={}
 def pre(module,args):state['c']=(module.lambdas[0]*args[0]).cpu();state['e']=(module.lambdas[1]*args[2]).cpu()
 def post(module,args,result):sources.append(torch.stack([state['c'],state['e'],result[0].cpu()]))
 handles=[model.transformer.h[8].register_forward_pre_hook(pre),model.transformer.h[8].attn.register_forward_hook(post)]
 try:m=measure(model,None,groups,['native'],None)
 finally:
  for h in handles:h.remove()
 v=expand(m['values'],mapping,6);old=torch.load(P/'TYPED_FACE_NATIVE8_FRESH_V1_ARTIFACT.pt',weights_only=True)['values'][0];anchor=float((v[0]-old).abs().max())
 fixtures=torch.load(P/'TYPED_FACE_MLP8_COUPLED_V1_ARTIFACT.pt',weights_only=True)['fixtures'];errors=[float((s.double().sum(0)-f['inputs']['post_attention8'].double()).norm()/f['inputs']['post_attention8'].norm()) for s,f in zip(sources,fixtures)]
 result={'pred_a':len(sources)==40 and max(errors)<=1e-6 and anchor<=1e-5 and bool(torch.isfinite(v).all()) and m['body_forwards']==40,'anchor_max_abs':anchor,'source_sum_max_relative':max(errors),'body_forwards':m['body_forwards'],'seconds':time.perf_counter()-start,'source_shas':binding,'scope':'Opened native context capture, no intervention. Ordered-product CPU checks separate.'}
 torch.save({'sources':sources,'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source_shas'}));signal.alarm(0)
if __name__=='__main__':main()
