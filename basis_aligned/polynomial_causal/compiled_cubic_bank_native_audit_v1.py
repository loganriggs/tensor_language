"""Native-weight cached-state replay of general source-bank projection.
Head9.8, original and after-head8-removal contexts; no new body forwards.
Checks implementation, not fidelity of the projected bank to the complete head.
"""
from pathlib import Path
import json,time,torch
import torch.nn.functional as F
from compiled_cubic_bank_head_v1 import compile_head,execute_pairs
from cubic_cluster_coordinates_v1 import components
from cubic_source_mixture_execute_v1 import execute as reference
from shared_cubic_source_projection_v1 import cross_factors
from folded_normalized_router_v1 import rotary,EPS
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'COMPILED_CUBIC_BANK_NATIVE_V1_AUDIT.json';assert not out.exists()
 bind=json.loads((P/'FOLDED_PRODUCER_CUBIC_NATIVE_V1_BINDING.json').read_text())['files'];sd=torch.load(next(k for k in bind if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 layer=9;head=8;prefix=f'transformer.h.{layer}.attn.';Q1,K1,Q2,K2=[sd[prefix+n+'.weight'].reshape(9,128,1152)[head] for n in ['c_q','c_k','c_q2','c_k2']]
 mu=float(sd[prefix+'lamb']);V=torch.cat(((1-mu)*sd[prefix+'c_v.weight'].double(),mu*sd['transformer.h.0.attn.c_v.weight'].double()),1).reshape(9,128,2304)[head];O=sd[prefix+'c_proj.weight'].double().reshape(1152,9,128)[:,head];C=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)['current_readers'].double()
 cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];lengths=torch.tensor([len(r['ids']) for r in rows]);ids=torch.zeros((len(rows),int(lengths.max())),dtype=torch.long)
 for i,row in enumerate(rows):ids[i,:len(row['ids'])]=torch.tensor(row['ids'])
 current=F.rms_norm(cache['r9'],(1152,),eps=EPS);emb=F.rms_norm(sd['transformer.wte.weight'][ids],(1152,),eps=EPS);lam=sd['transformer.h.0.lambdas'];first=F.rms_norm(lam[0]*emb+lam[1]*emb,(1152,),eps=EPS)
 assert current.shape[1:3]==ids.shape
 records=[]
 with torch.no_grad():
  for item in torch.load(P/'CUBIC_CLUSTER_NATIVE_V1_CHARTS.pt',weights_only=True):
   c,m=components(item['theta'],item['chart']);physical64=compile_head(c,m,Q1.double(),K1.double(),Q2.double(),K2.double(),V,O);physical32=compile_head(c,m,Q1,K1,Q2,K2,V,O);folded32=compile_head(c,m,Q1,K1,Q2,K2,V,C@O)
   for state in [0,1]:
    for length in sorted(set(lengths.tolist())):
     ix=torch.where(lengths==length)[0];q=current[state,ix,length-1];t=length-1
     for j in [0,7,t]:
      source=torch.cat((current[state,ix,j],first[ix,j]),1);r=rotary(t,128).T@rotary(j,128);ka=torch.cat((r@K1.double(),torch.zeros_like(K1,dtype=torch.float64)),1);kb=torch.cat((r@K2.double(),torch.zeros_like(K2,dtype=torch.float64)),1)
      factors=cross_factors(c,Q1.double()[None],ka[None],Q2.double()[None],kb[None],V[None],O[:,None]);expected=reference(q.double(),source.double(),c,m,factors).sum(2)[:,0];norm=torch.ones(len(q),dtype=torch.float64)
      for qm,km in [(Q1,K1),(Q2,K2)]:norm*=((q.double()@qm.double().T).square().mean(-1)+EPS)*((source[:,:1152].double()@km.double().T).square().mean(-1)+EPS)
      expected/=128**2*norm.sqrt()[:,None];p64=execute_pairs(q.double(),source.double(),r,physical64);p32=execute_pairs(q,source,r,physical32);f32=execute_pairs(q,source,r,folded32)
      error=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
      row=dict(arm=item['arm'],state=state,rows=len(ix),query_position=t,source_position=j,reference_norm=float(expected.norm()),fp64_reference_error=error(p64,expected),fp32_projection_error=error(p32,p64),consumer_commutation_error=error(f32,p32@C.T));records.append(row);print(json.dumps(row),flush=True)
 result=dict(pred_a=all(r['reference_norm']>1e-8 and r['fp64_reference_error']<=1e-8 for r in records),pred_b=all(r['fp32_projection_error']<=1e-4 for r in records),pred_c=all(r['consumer_commutation_error']<=1e-8 for r in records),rows=records,seconds=time.perf_counter()-tic,scope='Frozen earlier cluster banks; head9.8 native weights and48cached pristine/changed contexts at three position pairs. Mixed32/64 execution compared with independent64 projected-polynomial reference. Does not test approximation to the full native head, remaining FP32 normalization/rotary rounding, or language-model intervention effects.')
 out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
