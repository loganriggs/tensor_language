"""CPU-only direct block0–7 states for continuing the upstream fold."""
import hashlib,json,os,signal,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
sys.path[:0]=[str(ROOT/'basis_aligned/bilinear_quotient/ops'),str(P),str(ROOT)]
from regional_endpoint_batching_v1 import group_rows

@torch.no_grad()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 out=P/'CITY_PREFIX7_CPU_V1_RESULT.json';assert not out.exists()
 start=time.perf_counter();signal.alarm(180);torch.set_num_threads(2)
 rows=json.loads((P/'CITY_FULL_PILE_V2_ROWS.json').read_text())['rows'];groups,_=group_rows(rows);assert len(groups)==40
 from fastload import load_model_fast
 model=load_model_fast().eval();assert all(v.device.type=='cpu' for v in model.parameters())
 ids=torch.zeros(40,max(len(r['ids']) for r in groups),dtype=torch.long)
 for i,row in enumerate(groups):ids[i,:len(row['ids'])]=torch.tensor(row['ids'])
 x=F.rms_norm(model.transformer.wte(ids),(1152,));x0=x.clone();first=None;cache={}
 def pre7(module,args):
  cache['residual6']=args[0].clone();cache['mixed7']=(module.lambdas[0]*args[0]+module.lambdas[1]*args[2]).clone()
 def attnpre(module,args):cache['current7']=args[0].clone()
 def attnpost(module,args,result):cache['attention7']=result[0].clone()
 def mlppre(module,args):cache['normalized_mlp7']=args[0].clone()
 def mlppost(module,args,result):cache['mlp7']=result.clone()
 handles=[model.transformer.h[7].register_forward_pre_hook(pre7),model.transformer.h[7].attn.register_forward_pre_hook(attnpre),model.transformer.h[7].attn.register_forward_hook(attnpost),model.transformer.h[7].mlp.register_forward_pre_hook(mlppre),model.transformer.h[7].mlp.register_forward_hook(mlppost)]
 calls=0
 try:
  for block in model.transformer.h[:8]:x,first=block(x,first,x0);calls+=1
 finally:
  for h in handles:h.remove()
 cache.update({'initial':x0,'residual7':x,'first_values':first.reshape(40,ids.shape[1],1152)})
 gpu=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['fixtures'];sources=torch.load(P/'CITY_SOURCE7_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 reconstructed=torch.load(P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt',weights_only=True)['fixtures']
 tables=torch.load(P/'CITY_FULL_PILE_V2_TABLES.pt',weights_only=True);lookup={int(t):i for i,t in enumerate(tables['token_ids'])};lam=tables['lambdas8'];fixtures=[];records=[]
 def relative(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))
 for i,row in enumerate(groups):
  length=len(row['ids']);city=row['city_position'];f={k:v[i:i+1,:length].clone() for k,v in cache.items()};f['token_ids']=ids[i:i+1,:length].clone();f['city']=city;fixtures.append(f)
  local=torch.stack([lam[0]*f[k][:,city] for k in ['mixed7','attention7','mlp7']]+[lam[1]*f['initial'][:,city]])
  first_table=tables['first_table'][torch.tensor([lookup[t] for t in row['ids']])][None]
  records.append({'sequence':i,'residual7_error':relative(f['residual7'],gpu[i]['candidate_inputs']['residual7']),
                  'city_source_errors':[relative(a,b) for a,b in zip(local,sources[i]['sources'])],
                  'first_table_error':relative(f['first_values'],first_table),
                  'reconstructed_input_error':relative(f['normalized_mlp7'][:,city],reconstructed[i]['inputs']['normalized_mlp7_city'])})
 finite=all(bool(torch.isfinite(v).all()) for v in cache.values())
 r={'pred_a':max(x['residual7_error'] for x in records)<=1e-4,'pred_b':max(max(x['city_source_errors']) for x in records)<=1e-4,
    'pred_c':max(x['first_table_error'] for x in records)<=1e-4 and finite and len(fixtures)==40 and calls==8 and not torch.cuda.is_initialized(),
    'pred_d':max(x['reconstructed_input_error'] for x in records)<=1e-4,
    'sequences':records,'prefix_sequence_equivalents':40,'batched_block_calls':calls,'seconds':time.perf_counter()-start,
    'scope':'CPU-only native prefix; padded batch and device differ from GPU cache. No full-suffix, causal, fresh or composition claim.',
    'source_shas':{str(P/n):hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['CITY_PREFIX7_CPU_V1_PREREGISTRATION.md','capture_city_prefix7_cpu_v1.py']}}
 torch.save({'fixtures':fixtures},P/'CITY_PREFIX7_CPU_V1_ARTIFACT.pt');out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['sequences','source_shas']}));signal.alarm(0)
if __name__=='__main__':main()
