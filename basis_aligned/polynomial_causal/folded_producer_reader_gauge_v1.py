"""Exact downstream reader re-encoding versus Euclidean value-SVD stability."""
import torch,json,time
from pathlib import Path
from compiled_reading_head_v1 import execute_reading_head
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'FOLDED_PRODUCER_READER_GAUGE_V1_RESULT.json';assert not out.exists()
 binding=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files'];ck=next(k for k in binding if k.endswith('pytorch_model.bin'));state=torch.load(ck,weights_only=True,mmap=True,map_location='cpu')
 p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');a=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 C=p['current_readers'];layer=13;prefix=f'transformer.h.{layer}.attn.';scale=1.
 for j in range(layer+1,18):scale*=float(state[f'transformer.h.{j}.lambdas'][0])
 mix=float(state[prefix+'lamb']);fold=(scale*C@state[prefix+'c_proj.weight'].double()).reshape(4,9,128)[:,0]
 value=torch.cat([(1-mix)*state[prefix+'c_v.weight'].double().reshape(9,128,1152)[0],mix*state['transformer.h.0.attn.c_v.weight'].double().reshape(9,128,1152)[0]],1)
 M=fold@value;U,S,V=torch.linalg.svd(M,full_matrices=False);base=U[:,:1]@U[:,:1].T@M
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[])
 records=[]
 for axis in range(4):
  for factor in (.01,100.):
   scales=torch.ones(4,dtype=torch.float64);scales[axis]=factor
   u,s,v=torch.linalg.svd(scales[:,None]*M,full_matrices=False)
   selected=(u[:,:1]@u[:,:1].T@(scales[:,None]*M))/scales[:,None]
   gp=dict(p);gp['dual']=p['dual']/(scales[0]*scales[1]*scales[2:])[:,None]
   errors=[]
   for length in sorted(set(len(r['ids']) for r in rows)):
    ids=[i for i,r in enumerate(rows) if len(r['ids'])==length];query=a['current_states'][ids,length-1];qr=rotary(length-1,128)
    for pos in (0,length//2,length-1):
     current=a['current_states'][ids,pos];tokens=torch.tensor([rows[i]['ids'][pos] for i in ids]);rotation=(qr.T@rotary(pos,128)).float();f=current.double()@C.T+p['token_reads'][tokens]
     y=execute_reading_head(query,current,f,rotation,p);z=execute_reading_head(query,current,f*scales,rotation,gp)
     errors.append(float((z-y).norm()/y.norm()))
   records.append(dict(axis=axis,scale=factor,exact_function_replay_max=max(errors),leading_source_reader_cosine=float((v[0]@V[0]).abs()),selected_map_relative_change=float((selected-base).norm()/base.norm()),scaled_value_rank1_energy_fraction=float(s[0].square()/s.square().sum())))
 result=dict(records=records,seconds=time.perf_counter()-tic,scope='Exact diagonal re-encoding of the four downstream readings and compensating cubic dual. Whole branch identity checked on captured states; transformed value rank1 mapped back to original readings. No fitting, native wholehead changes, or behavioral rescue.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
