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
 torch.set_num_threads(2);tic=time.perf_counter();stem='STRUCTURED_PRODUCER_BANK_EFFECT_V1'
 out=P/(stem+'_RESULT.json');assert not out.exists()
 cache_result=json.loads((P/'STRUCTURED_PRODUCER_BANK_CACHE_V1_RESULT.json').read_text())
 assert all(cache_result[k] for k in ('pred_a','pred_b','pred_c'))
 a=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 c=torch.load(P/'STRUCTURED_PRODUCER_BANK_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[]);validate(rows)
 bank=torch.load(P/'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 bank_result=json.loads((P/'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_RESULT.json').read_text());assert all(bank_result[k] for k in ('pred_a','pred_b','pred_c'))
 heads=c['head_reads'];components=torch.einsum('hnti,hji->hntj',heads,bank['projectors'])
 contributions=torch.cat([heads,components,heads-components,c['reads'][:1]],0)
 branch=torch.zeros(83,96,1152);position_checks=[]
 for length in sorted(set(len(r['ids']) for r in rows)):
  ids=[i for i,r in enumerate(rows) if len(r['ids'])==length];donors=[i^1 for i in ids]
  query=a['current_states'][ids,length-1];qr=rotary(length-1,128)
  for pos in range(length):
   current=a['current_states'][ids,pos];tokens=torch.tensor([rows[i]['ids'][pos] for i in ids]);rotation=(qr.T@rotary(pos,128)).float()
   readings=current.double()@p['current_readers'].T+p['token_reads'][tokens]
   base=execute_reading_head(query,current,readings,rotation,p)
   position_checks.append(float((base-a['position_writes'][ids,pos]).norm()/a['position_writes'][ids,pos].norm().clamp_min(1e-30)))
   branch[0,ids]+=base
   for arm in range(82):
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
 baseline=margins(torch.zeros_like(branch[0]));effects=torch.stack([margins(branch[i+1]-branch[0])-baseline for i in range(82)])
 def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 baseline_error=rel(baseline,a['baseline_margins']);candidates=[]
 prior=torch.load(P/'CONSUMER_PULLBACK_EFFECT_CPU_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 prior_error=rel(effects[45],prior['effects'][2])
 instrument=max(position_checks)<=1e-5 and baseline_error<=1e-4 and prior_error<=1e-4
 for h in range(27):
  cells=[]
  for assignment in range(2):
   for family in range(2):
    ix=[i for i,row in enumerate(rows) if i//48==assignment and row['family']==family];assert len(ix)==24
    direction=torch.tensor([-1. if rows[i]['cue']=='British' else 1. for i in ix]);full=effects[h,ix,0];component=effects[27+h,ix,0];group=effects[81,ix,0]
    group_mean=float((group*direction).mean());component_mean=float((component*direction).mean());component_abs=float(component.abs().mean());unrelated=float(effects[27+h,ix,1].abs().mean())
    error=rel(component,full)
    cells.append(dict(assignment=assignment,family=family,fullhead_effect_norm=float(full.norm()),component_vs_head_error=error,group_mean_directed=group_mean,component_mean_directed=component_mean,component_group_fraction=component_mean/group_mean if group_mean>0 else None,component_positive_count=int((component*direction>0).sum()),component_meanabs=component_abs,unrelated_meanabs=unrelated,remainder_interaction_error=rel(component+effects[54+h,ix,0],full)))
  B=instrument and all(c['fullhead_effect_norm']>1e-6 and c['component_vs_head_error']<=.1 for c in cells)
  C=instrument and all(c['group_mean_directed']>0 and c['component_mean_directed']>=.1*c['group_mean_directed'] and c['component_positive_count']>=20 and c['unrelated_meanabs']<=.5*c['component_meanabs'] for c in cells)
  candidates.append(dict(layer=(8,9,13)[h//9],head=h%9,pred_b=B,pred_c=C,cells=cells))
 result={'pred_a':instrument,'baseline_replay':baseline_error,'no_edit_replay':max(position_checks),'head13_0_component_effect_replay':prior_error,'fidelity_pass_count':sum(c['pred_b'] for c in candidates),'transfer_specificity_pass_count':sum(c['pred_c'] for c in candidates),'both_pass_count':sum(c['pred_b'] and c['pred_c'] for c in candidates),'candidates':candidates,'seconds':time.perf_counter()-tic,'scope':'All27 frozen consumer-weighted value components, full native joint QK retained. Reused96contexts; passes are exploratory screens requiring fresh confirmation. No rank/hyperparameter search or text fitting.'}
 torch.save(dict(baseline=baseline,effects=effects,arm_layout='fullhead0:27,component27:54,remainder54:81,group81'),P/(stem+'_ARTIFACT.pt'))
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='candidates'},indent=2))
 for row in candidates:
  print(json.dumps({'layer':row['layer'],'head':row['head'],'B':row['pred_b'],'C':row['pred_c'],'fractions':[c['component_group_fraction'] for c in row['cells']],'errors':[c['component_vs_head_error'] for c in row['cells']]}))
if __name__=='__main__':main()
