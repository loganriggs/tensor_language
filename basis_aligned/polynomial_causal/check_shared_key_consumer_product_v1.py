"""Ranks frozen from weights; validate paired response on existing72prompts."""
from pathlib import Path
import json,torch
import torch.nn.functional as F
from head17_source_interface_v1 import CHECKPOINT
from shared_key_consumer_product_v1 import compile_program,scalar
from shared_key_reads_v1 import compile_program as full_compile,scalar as full_scalar
from contracted_qk_response_v1 import features,EPS
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True)
 rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
 native=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True)
 low=torch.load(P/'DIRECTIONAL_ROUTING_PREDICTOR_V1_TOP64.pt',weights_only=True)
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu');lam=sd['transformer.h.9.lambdas'].double();bias=sd['transformer.h.8.mlp.Down_bias'].double()
 B=native['key_basis'][1].double();ortherror=float((B.T@B-torch.eye(64)).norm());assert ortherror<1e-8
 full=full_compile(native,low,lam[0]);candidates={r:compile_program(native,low,lam[0],r) for r in [48]}
 size=lambda p:sum(v.numel() for v in p.values() if torch.is_tensor(v))
 baseline=[];predictions={r:[] for r in candidates}
 for i,row in enumerate(rows):
  n=len(row['ids']);ids=torch.tensor([row['ids']]);z=cache['z8'][0,i,:n][None].double();h=cache['raw9'][0,i,:n][None].double();a=cache['amplitude8'][0,i,:n][None]
  x0=F.rms_norm(F.embedding(ids,sd['transformer.wte.weight']),(1152,),eps=EPS).double();m=(h-lam[1]*x0)/lam[0]-z-bias
  rd,rho=features(z,h,m,a,full);base=full_scalar(h@full['readers'].T,h.square().mean(-1,keepdim=True)+EPS,full);baseline.append(float((full_scalar(rd,rho,full)-base)[0,-1]))
  for rank,(p,_) in candidates.items():
   rd,rho=features(z,h,m,a,p);base=scalar(h@p['readers'].T,h.square().mean(-1,keepdim=True)+EPS,p);predictions[rank].append(float((scalar(rd,rho,p)-base)[0,-1]))
 records=[];target=torch.tensor(baseline,dtype=torch.float64)
 for rank,(p,weight_error) in candidates.items():
  pred=torch.tensor(predictions[rank],dtype=torch.float64);cells=[]
  for family in sorted(set(r['family'] for r in rows)):
   mask=torch.tensor([r['family']==family for r in rows]);t=target[mask];e=pred[mask]-t
   cells.append(dict(family=family,response_error=float(e.norm()/t.norm()),reference_norm=float(t.norm())))
  saved=1-size(p)/size(full);records.append(dict(rank=rank,weight_error=weight_error,cells=cells,program_scalars=size(p),saved_fraction=saved,pred_a=weight_error<=.1,pred_b=all(c['response_error']<=.1 for c in cells),pred_c=saved>=.02))
 out={'records':records,'orthogonality_error':ortherror,'baseline_program_scalars':size(full),'scope':'Frozen best converged inside-product fit, not data fit. Paired scalar effects versus same rank64-transport program, existing72prompts. No fresh/native-logit/semantic-selectivity claim.'}
 (P/'SHARED_KEY_CONSUMER_PRODUCT_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
