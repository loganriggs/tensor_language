"""Opened CPU preflight, reconstructing MLP7 inputs from saved source captures."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from city_mlp7_integrated_v1 import execute
P=Path(__file__).resolve().parent
torch.set_num_threads(2)

def main():
 program=torch.load(P/'CITY_MLP7_READERS_V1_PROGRAM.pt',weights_only=True)
 head=torch.load(P/'extracted_circuits/typed_face_single_head_norm_v1/program.pt',weights_only=True)['head8']
 tables=torch.load(P/'CITY_FULL_PILE_V2_TABLES.pt',weights_only=True)
 lookup={int(t):i for i,t in enumerate(tables['token_ids'])};lam=tables['lambdas8'][0]
 captures=torch.load(P/'CITY_SOURCE7_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 refs=torch.load(P/'CITY_FULL_STRENGTH_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 errors=[];fixtures=[];eps=torch.finfo(torch.float32).eps
 for f,ref in zip(captures,refs,strict=True):
  source=f['sources'];current=f['current8'];city=ref['candidate_inputs']['city'];t=current.shape[1]
  # These are reconstructed states, not an exact capture of the native MLP input.
  z7=((source[0].double()+source[1].double())/lam.double()).float()
  normalized=F.rms_norm(z7,(1152,))
  inv=1/(10000**(torch.arange(0,128,2,dtype=torch.float32)/128));angles=torch.outer(torch.arange(t,dtype=torch.float32),inv)
  co,si=angles.cos().bfloat16().float(),angles.sin().bfloat16().float()
  queries=[];norms=[]
  for qn,kn in [('q1','k1'),('q2','k2')]:
   q=F.rms_norm(F.linear(current,head[qn]),(128,));a,b=q.chunk(2,-1)
   queries.append(torch.cat([a*co+b*si,-a*si+b*co],-1))
   k=F.linear(current,head[kn])[:,city].double();norms.append((k.square().mean(-1,keepdim=True)+eps).sqrt())
  token=int(ref['candidate_inputs']['token_ids'][0,city])
  inputs={'normalized_mlp7_city':normalized,'other_city_sources':source[[0,1,3]].double().sum(0),'lambda8':lam,
          'input_rms':(f['mixed8_city'].double().square().mean(-1,keepdim=True)+eps).sqrt(),'key_rms':torch.stack(norms,1),
          'rotated_queries':torch.stack(queries,2),'inherited_value':tables['first_table'][lookup[token],256:384][None],
          'city':city,'destination':ref['candidate_inputs']['destination']}
  got=execute(program,head,**inputs);expected=ref['inputs']['delta'].double()
  errors.append(float((got-expected).norm()/expected.norm()))
  fixtures.append({'inputs':inputs,'delta':got,'expected_native_delta':expected})
 result={'max_relative_write_error':max(errors),'per_sequence_errors':errors,'passes_inherited_diagnostic_bar':max(errors)<=1e-4,'scope':'Opened CPU preflight using reconstructed normalized MLP7 inputs. Native queries, mixed8 RMS and key RMS remain supplied. Separate direct native-input GPU certificate still required.'}
 torch.save({'fixtures':fixtures},P/'CITY_MLP7_INTEGRATED_V1_CPU_ARTIFACT.pt')
 (P/'CITY_MLP7_INTEGRATED_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='per_sequence_errors'}))
if __name__=='__main__':main()
