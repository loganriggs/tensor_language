from pathlib import Path
import json,torch
import torch.nn.functional as F
from head17_source_interface_v1 import CHECKPOINT
from shared_key_reads_v1 import compile_program as compile_full,expand
from shared_key_consumer_queryfold_v1 import compile_program as compile_small
from contracted_qk_response_v1 import features,scalar,EPS
from even_key_scalar_terms_v1 import terms
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True,mmap=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows']
 native=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True);low=torch.load(P/'DIRECTIONAL_ROUTING_PREDICTOR_V1_TOP64.pt',weights_only=True)
 sd=torch.load(CHECKPOINT,weights_only=True,mmap=True);lam=sd['transformer.h.9.lambdas'].double();bias=sd['transformer.h.8.mlp.Down_bias'].double();full=compile_full(native,low,lam[0]);small,_=compile_small(native,low,lam[0],48)
 samples=[];identity=0.
 for strength in [-1.,1.]:
  for i,row in enumerate(rows):
   ids=torch.tensor([row['ids']]);n=ids.shape[-1];z=cache['z8'][0,i,:n][None].double();h=cache['raw9'][0,i,:n][None].double();a=cache['amplitude8'][0,i,:n][None]*strength
   x0=F.rms_norm(F.embedding(ids,sd['transformer.wte.weight']),(1152,),eps=EPS).double();m=(h-lam[1]*x0)/lam[0]-z-bias;outputs=[]
   for p in [full,small]:
    pair=[]
    for amplitude in [torch.zeros_like(a),a]:
     rd,rho=features(z,h,m,amplitude,p);t=rd[...,512:-1]
     expanded=torch.cat([rd[...,:512],t@p['inside_adapters'][0].T,t@p['inside_adapters'][1].T,rd[...,-1:]],-1)
     tt=terms(expanded,rho);reference=scalar(expanded,rho);identity=max(identity,float((tt.sum(0)-reference).norm()/reference.norm().clamp_min(1e-30)));pair.append(tt[:,0,-1])
    outputs.append(pair[1]-pair[0])
   samples.append(dict(strength=strength,family=row['family'],reference=outputs[0],error=outputs[1]-outputs[0]))
 cells=[]
 for strength in [-1.,1.]:
  for family in sorted(set(r['family'] for r in rows)):
   selected=[s for s in samples if s['strength']==strength and s['family']==family];ref=torch.stack([s['reference'] for s in selected]).sum(-1);err=torch.stack([s['error'] for s in selected]);total=err.sum(-1);mixed=err[:,1];inside=err[:,2]
   cells.append(dict(strength=strength,family=family,response_error=float(total.norm()/ref.norm()),mixed_error_norm=float(mixed.norm()),inside_error_norm=float(inside.norm()),total_error_norm=float(total.norm()),mixed_inside_cosine=float(mixed@inside/(mixed.norm()*inside.norm())),full_full_error_norm=float(err[:,0].norm())))
 out={'pred_a':identity<=1e-10,'identity_error':identity,'cells':cells,'scope':'Exact algebraic attribution of existing conditional scalar compression error at±1. Mixed terms share full reads with inside reads; denominators identical across programs. Not independent causal interventions or new native validation.'}
 (P/'EVEN_KEY_ERROR_ATTRIBUTION_V1_CONTROL.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
