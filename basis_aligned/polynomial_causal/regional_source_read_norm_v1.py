"""Separate frozen source-feature readings from projected key RMS product, CPU."""
from pathlib import Path
import torch,json
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);out=P/'REGIONAL_SOURCE_READ_NORM_V1_RESULT.json';assert not out.exists()
 a=torch.load(P/'REGIONAL_SOURCE_POSITIONS_V1_ARTIFACT.pt',weights_only=True);p=torch.load(P/'MIXED_TOKEN_HEAD2_NATIVE_V1_PROGRAM.pt',weights_only=True)
 rows=sum([json.loads((P/(s+'_ROWS.json')).read_text())['rows'] for s in ('REGIONAL_COMPETING_CUES_V1','REGIONAL_CITY_ROLE_CROSSOVER_V1')],[])
 # CPU replay recomputes source hybrid per position, preserving frozen executor.
 from compiled_mixed_token_head_v1 import execute_mixed_token_head
 from folded_normalized_router_v1 import rotary
 base=torch.zeros_like(a['position_writes']);ports=torch.zeros(2,3,96,1152);eps=torch.finfo(torch.float32).eps;checks=[]
 for length in sorted(set(len(r['ids']) for r in rows)):
  ids=[i for i,r in enumerate(rows) if len(r['ids'])==length];q=a['current_states'][ids,length-1];qr=rotary(length-1,128)
  for pos in range(length):
   x=a['current_states'][ids,pos];tok=torch.tensor([rows[i]['ids'][pos] for i in ids]);r=(qr.T@rotary(pos,128)).float()
   b=execute_mixed_token_head(q,x,tok,r,p);base[ids,pos]=b
   def norm(x):return (((x@p['k1'].T).square().mean(-1)+eps)*((x@p['k2'].T).square().mean(-1)+eps)).sqrt()
   n=norm(x)
   for k,bit in enumerate((1,2)):
    donor=[i^bit for i in ids];dx=a['current_states'][donor,pos];dt=torch.tensor([rows[i]['ids'][pos] for i in donor]);dn=norm(dx)
    both=execute_mixed_token_head(q,dx,dt,r,p)
    features=both*(dn/n)[:,None];normalizer=b*(n/dn)[:,None]
    ports[k,0,ids]+=both;ports[k,1,ids]+=features;ports[k,2,ids]+=normalizer
    checks.append(float((features*(n/dn)[:,None]-both).norm()/both.norm().clamp_min(1e-30)))
 prev=torch.load(P/'REGIONAL_QUERY_SOURCE_ACCOUNTING_V1_ARTIFACT.pt',weights_only=True)
 result=dict(gate_ratio_roundtrip=max(checks),source_hybrid_replay=float((ports[:,0]-prev['hybrid_writes'][:,1]).norm()/prev['hybrid_writes'][:,1].norm()),scope='Source-only fixed-query intervention decomposed into four feature readings versus product of two native projected key RMS factors. No separate QK1/QK2 task assignment. Native effects pending.')
 torch.save(dict(base_write=base.sum(1),port_writes=ports),P/'REGIONAL_SOURCE_READ_NORM_V1_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
