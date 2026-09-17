"""CPU diagnostic for a one-head normalization candidate; no causal adoption."""
from pathlib import Path
import sys,json,hashlib
import torch
import torch.nn.functional as F
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2)
 folder=P/'extracted_circuits/typed_face_reduced_fused_v1';sys.path.insert(0,str(folder))
 import execute as exact
 import typed_face_reduced_norm_variants_v1 as frozen
 import typed_face_single_head_norm_v1 as candidate
 from attention8_context_channels_v1 import channels as all_channels
 from attention8_single_head_channels_v1 import channels as one_channel
 p={k:torch.load(folder/f,weights_only=True) for k,f in [('local','local_program.pt'),('context','context_program.pt'),('reentry','reentry_program.pt')]}
 fixtures=torch.load(P/'TYPED_FACE_REDUCED_RESIDUAL7_V1_ARTIFACT.pt',weights_only=True)['fixtures']
 from regional_endpoint_batching_v1 import group_rows
 groups,_=group_rows(json.loads((P/'HEAD2_MLP8_CROSS_FRESH_V1_ROWS.json').read_text())['rows'])
 stats={};channel_errors=[];zeros=[]
 for row,f in zip(groups,fixtures):
  x=f['inputs'];e=p['reentry'];lookup={int(t):i for i,t in enumerate(e['token_ids'].tolist())}
  idx=torch.tensor([[lookup[int(t)] for t in r] for r in x['token_ids'].tolist()])
  mixed=e['lambdas8'][0]*x['residual7']+e['lambdas8'][1]*F.embedding(idx,e['initial_table']).to(x['residual7'].dtype)
  normalized=F.rms_norm(mixed,(1152,));a=all_channels(p['context'],normalized,x['token_ids'])[:,:,2];b=one_channel(p['context'],normalized,x['token_ids'])[:,:,0]
  channel_errors.append(float((a-b).norm()/a.norm()))
  got=candidate.execute(p,**x)
  for name,ref in [('exact',exact.execute(p,**x)),('full_context_frozen',frozen.execute(p,**x,mode='freeze_denominator'))]:
   acc=stats.setdefault(row['variant'],{}).setdefault(name,[0.,0.,0.,0.]);acc[0]+=float((got-ref).square().sum());acc[1]+=float(ref.square().sum());acc[2]+=float(got.square().sum());acc[3]+=float((got*ref).sum())
  zeros.append(float(candidate.execute(p,**x,strength=0).abs().max()))
 result={'scope':'Local write diagnostics on 40 opened sequences / 20 cells. No model suffix evaluated; no fresh or selectivity claim. Single-head norm is an approximation, not exact elimination of context.', 'head_channel_max_relative_error':max(channel_errors),'zero_strength_max':max(zeros),'instrument_pass':max(channel_errors)<1e-5 and max(zeros)==0,'families':{fam:{name:{'relative_vector_error':(v[0]/v[1])**.5,'norm_ratio':(v[2]/v[1])**.5,'aligned_fraction':v[3]/v[1]} for name,v in refs.items()} for fam,refs in stats.items()},'source_sha256':{name:hashlib.sha256((P/name).read_bytes()).hexdigest() for name in ['typed_face_single_head_norm_v1.py','attention8_single_head_channels_v1.py','check_single_head_norm_v1_cpu.py']}}
 assert result['instrument_pass'];(P/'SINGLE_HEAD_NORMALIZATION_V1_CPU_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
