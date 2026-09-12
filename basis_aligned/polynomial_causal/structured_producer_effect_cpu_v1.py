"""Frozen native producer-to-source-reader edge test. Bars in preregistration."""
from pathlib import Path
import torch,json,time
import torch.nn.functional as F
from compiled_reading_head_v1 import execute_reading_head
from folded_normalized_router_v1 import rotary
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();stem='STRUCTURED_PRODUCER_EFFECT_CPU_V1'
 out=P/(stem+'_RESULT.json');assert not out.exists()
 cache_result=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_RESULT.json').read_text())
 assert all(cache_result[k] for k in ('pred_a','pred_b','pred_c'))
 a=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 c=torch.load(P/'STRUCTURED_PRODUCER_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[]);validate(rows)
 contributions=torch.cat([c['reads'],(c['reads'][1]-c['reads'][2])[None]],0)
 branch=torch.zeros(5,96,1152);position_checks=[]
 for length in sorted(set(len(r['ids']) for r in rows)):
  ids=[i for i,r in enumerate(rows) if len(r['ids'])==length];donors=[i^1 for i in ids]
  query=a['current_states'][ids,length-1];qr=rotary(length-1,128)
  for pos in range(length):
   current=a['current_states'][ids,pos];tokens=torch.tensor([rows[i]['ids'][pos] for i in ids]);rotation=(qr.T@rotary(pos,128)).float()
   readings=current.double()@p['current_readers'].T+p['token_reads'][tokens]
   base=execute_reading_head(query,current,readings,rotation,p)
   position_checks.append(float((base-a['position_writes'][ids,pos]).norm()/a['position_writes'][ids,pos].norm().clamp_min(1e-30)))
   branch[0,ids]+=base
   for arm in range(4):
    change=(contributions[arm,donors,pos]-contributions[arm,ids,pos])/c['rho'][ids,pos]
    branch[arm+1,ids]+=execute_reading_head(query,current,readings+change,rotation,p)
 binding=json.loads((P/'REGIONAL_QUERY_SOURCE_EFFECT_V1_BINDING.json').read_text())['files'];checkpoint=next(k for k in binding if k.endswith('pytorch_model.bin'))
 state=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
 ids=torch.tensor([[r['uk_id'],r['us_id'],*r['control_ids']] for r in rows]);vocab,inverse=torch.unique(ids,sorted=True,return_inverse=True)
 u=state['lm_head.weight'][vocab].float();l,r,d=[state['transformer.h.17.mlp.'+k+'.weight'].float() for k in ('Left','Right','Down')];bias=state['transformer.h.17.mlp.Down_bias'].float()
 def margins(delta):
  z=c['pre']+delta;x=F.rms_norm(z,(1152,));h=z+F.linear(F.linear(x,l)*F.linear(x,r),d,bias)
  logits=30*torch.tanh(F.linear(F.rms_norm(h,(1152,)),u)/30);picked=logits.gather(1,inverse)
  return torch.stack([picked[:,0]-picked[:,1],picked[:,2]-picked[:,3]],1).double()
 baseline=margins(torch.zeros_like(branch[0]));effects=torch.stack([margins(branch[i+1]-branch[0])-baseline for i in range(4)])
 def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 baseline_error=rel(baseline,a['baseline_margins']);cells=[]
 for assignment in range(2):
  for family in range(2):
   ix=[i for i,row in enumerate(rows) if i//48==assignment and row['family']==family];assert len(ix)==24
   direction=torch.tensor([-1. if rows[i]['cue']=='British' else 1. for i in ix]);transfer=effects[:,ix,0]*direction
   full=effects[1,ix,0];rank=effects[2,ix,0]
   cells.append(dict(assignment=assignment,family=family,head_regional_effect_norm=float(full.norm()),rank1_vs_head_relative_error=rel(rank,full),mean_directed_regional_transfer=transfer.mean(1).tolist(),positive_directed_count=(transfer>0).sum(1).tolist(),meanabs_regional=effects[:,ix,0].abs().mean(1).tolist(),meanabs_unrelated=effects[:,ix,1].abs().mean(1).tolist(),rank1_plus_complement_vs_head_relative_error=rel(rank+effects[3,ix,0],full)))
 instrument=max(position_checks)<=1e-5 and baseline_error<=1e-4
 fidelity=instrument and all(c['head_regional_effect_norm']>1e-6 and c['rank1_vs_head_relative_error']<=.1 for c in cells)
 selective=instrument and all(c['mean_directed_regional_transfer'][0]>0 and c['mean_directed_regional_transfer'][2]>=.1*c['mean_directed_regional_transfer'][0] and c['positive_directed_count'][2]>=20 and c['meanabs_unrelated'][2]<=.5*c['meanabs_regional'][2] for c in cells)
 result={'pred_a':instrument,'pred_b':fidelity,'pred_c':selective,'no_edit_position_replay_max':max(position_checks),'baseline_replay':baseline_error,'cells':cells,'arm_names':['three_producer_group','head13_0','head13_0_value_rank1','head13_0_value_complement'],'seconds':time.perf_counter()-tic,'scope':'Frozen conditional producer contribution swaps into regional four-reader interface. Query, native key norms, other states, first-token lookup fixed. No text fitting, no whole module replacement or independent extraction.'}
 torch.save(dict(branch_writes=branch,baseline=baseline,effects=effects),P/(stem+'_ARTIFACT.pt'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
