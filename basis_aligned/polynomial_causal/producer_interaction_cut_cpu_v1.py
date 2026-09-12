"""Exact ten-path source-group cubic cut; fixed registered mixed-path tests."""
from pathlib import Path
import itertools,json,time,torch
import torch.nn.functional as F
from compiled_feature_head_v1 import execute_feature_head
from compiled_reading_head_v1 import execute_reading_head
from folded_normalized_router_v1 import rotary
from regional_cue_row_check_v1 import validate
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();stem='PRODUCER_INTERACTION_CUT_CPU_V1';out=P/(stem+'_RESULT.json');assert not out.exists()
 a=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');c=torch.load(P/'STRUCTURED_PRODUCER_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True,map_location='cpu')
 assert all(json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_RESULT.json').read_text())[k] for k in ('pred_a','pred_b','pred_c'))
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[]);validate(rows)
 paths=list(itertools.combinations_with_replacement(range(3),3));ordered=list(itertools.product(range(3),repeat=3));path_ids=[paths.index(tuple(sorted(t))) for t in ordered]
 terms=torch.zeros(2,10,96,1152);reference=torch.zeros(2,96,1152);feature_checks=[]
 for length in sorted(set(len(r['ids']) for r in rows)):
  ids=[i for i,r in enumerate(rows) if len(r['ids'])==length];donor=[i^1 for i in ids];query=a['current_states'][ids,length-1];qr=rotary(length-1,128)
  for pos in range(length):
   x=a['current_states'][ids,pos];token=torch.tensor([rows[i]['ids'][pos] for i in ids]);rotation=(qr.T@rotary(pos,128)).float();f=x.double()@p['current_readers'].T+p['token_reads'][token];rho=c['rho'][ids,pos];background=f-c['reads'][0,ids,pos]/rho
   for variant,source in enumerate((ids,donor)):
    head=c['reads'][1,source,pos]/rho;other=(c['reads'][0,source,pos]-c['reads'][1,source,pos])/rho
    groups=torch.stack([background,head,other]);z=torch.zeros(10,len(ids),2,dtype=torch.float64)
    for (i,j,k),path in zip(ordered,path_ids):z[path]+=groups[i,:,0,None]*groups[j,:,1,None]*groups[k,:,2:]
    total=groups.sum(0);ref=execute_reading_head(query,x,total,rotation,p);reference[variant,ids]+=ref
    combined=execute_feature_head(query,x,z.sum(0),rotation,p)
    feature_checks.append(float((combined-ref).norm()/ref.norm().clamp_min(1e-30)))
    for path in range(10):terms[variant,path,ids]+=execute_feature_head(query,x,z[path],rotation,p)
 deltas=terms[1]-terms[0];full=reference[1]-reference[0]
 binding=json.loads((P/'STRUCTURED_PRODUCER_CACHE_V1_BINDING.json').read_text())['files'];state=torch.load(next(k for k in binding if k.endswith('pytorch_model.bin')),weights_only=True,mmap=True,map_location='cpu')
 ids=torch.tensor([[r['uk_id'],r['us_id'],*r['control_ids']] for r in rows]);vocab,inverse=torch.unique(ids,sorted=True,return_inverse=True);u=state['lm_head.weight'][vocab].float()
 l,r,d=[state['transformer.h.17.mlp.'+k+'.weight'].float() for k in ('Left','Right','Down')];bias=state['transformer.h.17.mlp.Down_bias'].float()
 def margins(delta):
  z=c['pre']+delta;x=F.rms_norm(z,(1152,));h=z+F.linear(F.linear(x,l)*F.linear(x,r),d,bias);logits=30*torch.tanh(F.linear(F.rms_norm(h,(1152,)),u)/30);v=logits.gather(1,inverse);return torch.stack([v[:,0]-v[:,1],v[:,2]-v[:,3]],1).double()
 baseline=margins(torch.zeros_like(full));effect=margins(full)-baseline
 individual=torch.stack([margins(delta)-baseline for delta in deltas])
 mixed=[i for i,t in enumerate(paths) if len(set(t))>1];cross=[i for i,t in enumerate(paths) if 1 in t and 2 in t];linear=[i for i,t in enumerate(paths) if t.count(0)==2]
 grouped=torch.stack([margins(deltas[indices].sum(0))-baseline for indices in (mixed,cross,linear)])
 def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 prior=torch.load(P/'STRUCTURED_PRODUCER_EFFECT_CPU_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')
 checks=dict(feature_replay=max(feature_checks),partition_write_replay=rel(terms.sum(1),reference),full_group_effect_replay=rel(effect,prior['effects'][0]),zero_background_delta=float(deltas[0].norm()))
 cells=[]
 for assignment in range(2):
  for family in range(2):
   ix=[i for i,row in enumerate(rows) if i//48==assignment and row['family']==family];direction=torch.tensor([-1. if rows[i]['cue']=='British' else 1. for i in ix])
   cells.append(dict(assignment=assignment,family=family,mixed_vs_full_error=rel(grouped[0,ix,0],effect[ix,0]),cross_producer_to_full_effect_norm=float(grouped[1,ix,0].norm()/effect[ix,0].norm()),cross_producer_mean_directed=float((grouped[1,ix,0]*direction).mean()),linear_in_producers_vs_full_error=rel(grouped[2,ix,0],effect[ix,0]),individual_mean_directed=(individual[:,ix,0]*direction).mean(1).tolist(),full_mean_directed=float((effect[ix,0]*direction).mean()),sum_individual_vs_full_effect_error=rel(individual[:,ix,0].sum(0),effect[ix,0])))
 A=max(checks['feature_replay'],checks['partition_write_replay'])<=1e-5 and checks['full_group_effect_replay']<=1e-4
 result={'pred_a':A,'pred_b':A and all(c['mixed_vs_full_error']<=.1 for c in cells),'pred_c':A and all(c['cross_producer_to_full_effect_norm']>=.1 and c['cross_producer_mean_directed']>0 for c in cells),'checks':checks,'paths':paths,'group_names':['background','head13_0','other_selected_producers'],'cells':cells,'seconds':time.perf_counter()-tic,'scope':'Exact ten-path grouped cubic expansion of frozen regional reader interface. Donor producers and recipient background/query/key gates. Individual path native effects need not add; full write partition does. No unsupervised factor fit or wholemodel causal closure.'}
 torch.save(dict(path_writes=terms,full_effect=effect,path_effects=individual,grouped_effects=grouped),P/(stem+'_ARTIFACT.pt'));out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
