"""Exact two-port write accounting on saved competing-city native states, CPU only."""
from pathlib import Path
import torch,json,time
from compiled_mixed_token_head_v1 import execute_mixed_token_head
from folded_normalized_router_v1 import rotary
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'REGIONAL_QUERY_SOURCE_ACCOUNTING_V1_RESULT.json';assert not out.exists()
 a=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True);p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[])
 base=torch.zeros_like(a['position_writes']);hybrid=torch.zeros(2,2,*base.shape);before=[]
 for length in sorted(set(len(r['ids']) for r in rows)):
  ids=[i for i,r in enumerate(rows) if len(r['ids'])==length];q=a['current_states'][ids,length-1];qr=rotary(length-1,128)
  for pos in range(length):
   rotation=(qr.T@rotary(pos,128)).float();current=a['current_states'][ids,pos];tokens=torch.tensor([rows[i]['ids'][pos] for i in ids])
   base[ids,pos]=execute_mixed_token_head(q,current,tokens,rotation,p)
   for k,bit in enumerate((1,2)):
    donor=[i^bit for i in ids];dq=a['current_states'][donor,length-1];dc=a['current_states'][donor,pos];dt=torch.tensor([rows[i]['ids'][pos] for i in donor])
    hybrid[k,0,ids,pos]=execute_mixed_token_head(dq,current,tokens,rotation,p)
    hybrid[k,1,ids,pos]=execute_mixed_token_head(q,dc,dt,rotation,p)
    for j,i in enumerate(ids):
     if pos<a['city_positions'][i][k]:before.append(float((current[j]-dc[j]).abs().max()))
 def rel(x,y):return float((x-y).norm()/y.norm().clamp_min(1e-30))
 cells=[];qs=[]
 for k,name in enumerate(('editor','tourist')):
  donor=torch.arange(96)^ (1 if k==0 else 2);end=base[donor];delta=end-base
  query=.5*(hybrid[k,0]-base+end-hybrid[k,1]);source=.5*(hybrid[k,1]-base+end-hybrid[k,0]);qs.append(torch.stack([query.sum(1),source.sum(1)],1))
  for assignment in range(2):
   for family in range(2):
    ids=list(range(assignment*48+family*24,assignment*48+family*24+24));d=delta[ids].sum(1);q=query[ids].sum(1);s=source[ids].sum(1)
    cells.append(dict(cue=name,assignment=assignment,family=family,partition_error=rel(q+s,d),query_norm_relative=float(q.norm()/d.norm()),source_norm_relative=float(s.norm()/d.norm()),query_source_cosine=float((q*s).sum()/(q.norm()*s.norm())),source_only_error=rel(hybrid[k,1,ids].sum(1)-base[ids].sum(1),d),query_only_error=rel(hybrid[k,0,ids].sum(1)-base[ids].sum(1),d)))
 result=dict(cpu_gpu_replay=rel(base,a['position_writes']),before_cue_state_max_difference=max(before),cells=cells,seconds=time.perf_counter()-tic,scope='Exact two-port Shapley accounting of physical component write changes on96cached native contexts. Query/source norms are not additive fractions or native logit mediation. Source port includes current-state readers, keys and token lookup; query port includes both QK factors and query norms.')
 torch.save(dict(query_source_writes=torch.stack(qs),base_write=base.sum(1),hybrid_writes=hybrid.sum(3)),P/'REGIONAL_QUERY_SOURCE_ACCOUNTING_V1_ARTIFACT.pt')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
