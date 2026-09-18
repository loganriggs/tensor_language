"""Installed equivalence of reduced-input mediator on opened fresh fixtures."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows,expand
from regional_paired_write_runtime_v4 import CONTROL_PAIRS
STEM='CITY_MLP8_VALUE_EXTRACTED_INSTALLED_V1'
@torch.no_grad()
def main():
 binding=json.loads((P/(STEM+'_BINDING.json')).read_text())['files'];assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in binding.items())
 rows=json.loads((P/'CITY_VALUE_MEDIATION_FRESH_V1_ROWS.json').read_text())['rows'];groups,mapping=group_rows(rows)
 if os.environ.get('BQLIB_DRYRUN') or os.environ.get('BQLIB_NO_MODEL'):print('80sequence equivalents;36block calls;CPU extracted mediator');return
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);start=time.perf_counter();signal.alarm(180);out=P/(STEM+'_RESULT.json');assert not out.exists()
 from fastload import load_model_fast
 model=load_model_fast().eval();fixtures=torch.load(P/'CITY_MLP8_VALUE_EXTRACTED_V1_ARTIFACT.pt',weights_only=True)['fixtures'];old=torch.load(P/'CITY_VALUE_MEDIATION_FRESH_V1_ARTIFACT.pt',weights_only=True)['values'][[0,3]]
 ids=torch.tensor([r['ids'] for r in groups]);edit=torch.cat([f['inputs']['delta'] for f in fixtures]).float();correction=torch.cat([f['delta'] for f in fixtures]).float();state={'arm':0}
 def post8(module,args,out):return (out[0]+edit,out[1]) if state['arm'] else out
 h=model.transformer.h[8].attn.register_forward_hook(post8);a=model.transformer.h[9].attn;original=a.squared_attention
 def value(module,q,k,v,q2,k2):
  if state['arm']:v=v.clone();v[:,:,8]-=correction
  return original(q,k,v,q2,k2)
 a.squared_attention=MethodType(value,a);values=torch.zeros(2,40,10,dtype=torch.float64);calls=0
 try:
  for arm in range(2):
   state['arm']=arm;x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x;first=None
   for block in model.transformer.h:x,first=block(x,first,x0);calls+=1
   scores=30*torch.tanh(model.lm_head(F.rms_norm(x[:,-1],(1152,)))/30)
   for i,row in enumerate(groups):
    for j,(left,right) in enumerate(row['endpoint_pairs']+CONTROL_PAIRS):values[arm,i,j]=scores[i,left]-scores[i,right]
 finally:h.remove();a.squared_attention=original
 v=expand(values,mapping,6);diff=v-old;effect=v[1]-v[0];old_effect=old[1]-old[0];errors=((effect-old_effect).norm(dim=0)/old_effect.norm(dim=0)).tolist()
 outside=float(correction[edit.abs().sum(-1)==0].abs().max())
 r={'pred_a':float(diff.abs().max())<=1e-4 and float(diff.norm()/old.norm())<=1e-5,'pred_b':max(errors)<=1e-4,'pred_c':outside==0 and bool(torch.isfinite(v).all()) and calls==36 and not torch.cuda.is_initialized(),'readout_effect_errors':errors,'replay_max_abs':float(diff.abs().max()),'replay_relative':float(diff.norm()/old.norm()),'outside':outside,'body_forwards':80,'batched_block_calls':calls,'seconds':time.perf_counter()-start,'scope':'Opened equivalence: exact native cityswap plus reduced-interface MLP8value correction. Native z8 and editedrho9 remain inputs; fullsuffix external.','source_shas':binding}
 torch.save({'values':v},P/(STEM+'_ARTIFACT.pt'));out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));signal.alarm(0)
if __name__=='__main__':main()
