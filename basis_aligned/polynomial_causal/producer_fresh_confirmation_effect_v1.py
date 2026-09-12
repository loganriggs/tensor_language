"""Fresh fixed producer components, shared output, joint and removal effects."""
from pathlib import Path
import torch,json,time
import torch.nn.functional as F
from compiled_reading_head_v1 import execute_reading_head
from folded_normalized_router_v1 import rotary
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();stem='PRODUCER_FRESH_CONFIRMATION_EFFECT_V1';out=P/(stem+'_RESULT.json');assert not out.exists()
 native=json.loads((P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_RESULT.json').read_text());assert native['pred_a'] and native['pred_c']
 c=torch.load(P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');bank=torch.load(P/'STRUCTURED_PRODUCER_BANK_WEIGHTS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');merged=torch.load(P/'STRUCTURED_PRODUCER_SHARED_OUTPUT_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];validate(rows);ix=[2,17];heads=c['head_reads'][ix]
 components=torch.einsum('hnti,hji->hntj',heads,bank['projectors'][ix]);merge=torch.einsum('hnti,hji->hntj',heads,merged['projectors'])
 contributions=torch.cat([c['group_reads'][None],heads,components,components.sum(0,keepdim=True),merge,merge.sum(0,keepdim=True),components,components.sum(0,keepdim=True)])
 assert contributions.shape==(12,48,22,4)
 writes=torch.zeros(13,48,1152);checks=[]
 for length in sorted(set(len(r['ids']) for r in rows)):
  ids=[i for i,r in enumerate(rows) if len(r['ids'])==length];donors=[i^1 for i in ids];query=c['current_states'][ids,length-1];qr=rotary(length-1,128)
  for pos in range(length):
   current=c['current_states'][ids,pos];tokens=torch.tensor([rows[i]['ids'][pos] for i in ids]);rotation=(qr.T@rotary(pos,128)).float();f=current.double()@p['current_readers'].T+p['token_reads'][tokens]
   base=execute_reading_head(query,current,f,rotation,p);writes[0,ids]+=base;ref=c['position_writes'][ids,pos];checks.append(float((base-ref).norm()/ref.norm().clamp_min(1e-30)))
   for arm in range(12):
    delta=contributions[arm,donors,pos]-contributions[arm,ids,pos] if arm<9 else -contributions[arm,ids,pos]
    writes[arm+1,ids]+=execute_reading_head(query,current,f+delta/c['rho'][ids,pos],rotation,p)
 binding=json.loads((P/'PRODUCER_FRESH_CONFIRMATION_CACHE_V1_BINDING.json').read_text())['files'];state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 ids=torch.tensor([[r['uk_id'],r['us_id'],*r['control_ids'],*r['newline_control_ids']] for r in rows]);vocab,inverse=torch.unique(ids,sorted=True,return_inverse=True);u=state['lm_head.weight'][vocab].float();l,r,d=[state['transformer.h.17.mlp.'+k+'.weight'].float() for k in ('Left','Right','Down')];bias=state['transformer.h.17.mlp.Down_bias'].float()
 def margins(delta):
  z=c['pre']+delta;x=F.rms_norm(z,(1152,));h=z+F.linear(F.linear(x,l)*F.linear(x,r),d,bias);logits=30*torch.tanh(F.linear(F.rms_norm(h,(1152,)),u)/30);v=logits.gather(1,inverse);return torch.stack([v[:,0]-v[:,1],v[:,2]-v[:,3],v[:,4]-v[:,5]],1).double()
 baseline=margins(torch.zeros_like(writes[0]));effects=torch.stack([margins(writes[i+1]-writes[0])-baseline for i in range(12)])
 def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 baseline_error=rel(baseline,c['baseline_margins']);cells=[]
 for family in range(2):
  ids=[i for i,row in enumerate(rows) if row['family']==family];direction=torch.tensor([-1. if rows[i]['cue']=='British' else 1. for i in ids]);uk=[i for i in ids if rows[i]['cue']=='British'];us=[i^1 for i in uk]
  means=(effects[:,ids,0]*direction).mean(1);counts=(effects[:,ids,0]*direction>0).sum(1);meanabs=effects[:,ids,0].abs().mean(1);unrelated=effects[:,ids,1].abs().mean(1)
  fids=[rel(effects[3+k,ids,0],effects[1+k,ids,0]) for k in range(2)];merge_fids=[rel(effects[6+k,ids,0],effects[3+k,ids,0]) for k in range(3)]
  fidelity=all(e<=.1 and float(effects[1+k,ids,0].norm())>1e-6 for k,e in enumerate(fids))
  selectivity=means[0]>0 and all(means[3+k]>=.1*means[0] and counts[3+k]>=20 and unrelated[3+k]<=.5*meanabs[3+k] for k in range(2)) and means[5]>=.8*means[0] and counts[5]>=20
  cells.append(dict(family=family,individual_fidelity_pass=bool(fidelity),transfer_and_joint_pass=bool(selectivity),merge_pass=max(merge_fids)<=.1,component_vs_head_errors=fids,merged_vs_unmerged_errors=merge_fids,mean_directed_by_arm=means.tolist(),positive_count_by_arm=counts.tolist(),meanabs_regional=meanabs.tolist(),meanabs_unrelated=unrelated.tolist(),newline_meanabs_by_arm=effects[:,ids,2].abs().mean(1).tolist(),component_joint_group_transfer_fraction=float(means[5]/means[0]) if means[0]>0 else None,individual_sum_vs_joint_error=rel(effects[3,ids,0]+effects[4,ids,0],effects[5,ids,0]),removal_cue_contrast_change=(effects[9:12,uk,0]-effects[9:12,us,0]).mean(1).tolist()))
 A=max(checks)<=1e-5 and baseline_error<=1e-4;B=A and native['pred_b'] and all(c['individual_fidelity_pass'] and c['transfer_and_joint_pass'] for c in cells);C=B and all(c['merge_pass'] for c in cells)
 result={'pred_a':A,'pred_b':B,'pred_c':C,'native_capability':native['pred_b'],'capability_cells':native['capability_cells'],'noedit_replay_max':max(checks),'baseline_replay':baseline_error,'cells':cells,'arms':['group_swap','head8_2_swap','head9_8_swap','component8_2_swap','component9_8_swap','joint_components_swap','merged8_2_swap','merged9_8_swap','joint_merged_swap','remove_component8_2','remove_component9_8','remove_joint_components'],'seconds':time.perf_counter()-tic,'scope':'Frozen fresh lexical/template confirmation of conditional producer-to-regional-readings edges. Complete native query/routing/background retained. Newline endpoint here is descriptive without a newline-capability panel. No text fitting or wholehead removal claim.'}
 torch.save(dict(baseline=baseline,effects=effects,branch_writes=writes),P/(stem+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
