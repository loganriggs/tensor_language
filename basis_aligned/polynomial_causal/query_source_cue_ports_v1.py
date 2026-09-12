from pathlib import Path
import json,time,torch
import torch.nn.functional as F
from regional_even_query_source_v1 import scalar
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];p=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True);native=F.rms_norm(cache['raw9'][0],(1152,));corners=[];checks=[];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 for i in range(0,72,2):
  n=len(rows[i]['ids']);u=native[i:i+1,:n];a=native[i+1:i+2,:n];ids=torch.tensor([rows[i]['ids']]);values=[]
  for q,s in [(u,u),(a,u),(u,a),(a,a)]:values.append(scalar(q,s,ids,p,1))
  checks.extend([rel(values[0],cache['scalar'][0,i,:n][None]),rel(values[3],cache['scalar'][0,i+1,:n][None])]);corners.append(torch.stack([v[0,-1] for v in values]))
 x=torch.stack(corners);target=x[:,0]-x[:,3];query=((x[:,0]-x[:,1])+(x[:,2]-x[:,3]))/2;source=((x[:,0]-x[:,2])+(x[:,1]-x[:,3]))/2;interaction=x[:,0]-x[:,1]-x[:,2]+x[:,3];records=[]
 for family in range(3):
  ix=[i//2 for i in range(0,72,2) if rows[i]['family']==family];d=target[ix];q=query[ix];s=source[ix];records.append(dict(family=family,name=rows[ix[0]*2]['family_name'],target_norm=float(d.norm()),query_aligned_fraction=float((q@d)/d.square().sum()),source_aligned_fraction=float((s@d)/d.square().sum()),query_only_error=rel(x[ix,2]-x[ix,3],d),source_only_error=rel(x[ix,1]-x[ix,3],d),interaction_norm_over_target=float(interaction[ix].norm()/d.norm())))
 accounting=rel(query+source,target);A=max(checks)<=1e-5 and accounting<=1e-10;result={'pred_a':A,'pred_b':A and all(r['query_only_error']<=.2 for r in records),'pred_c':A and all(r['source_only_error']<=.2 for r in records),'max_native_anchor_error':max(checks),'accounting_error':accounting,'records':records,'seconds':time.perf_counter()-tic,'scope':'36paired nativecue contexts from prior shortpanel. Head9.8 even scalar only, factorial query vs sourceKV hybrids; symmetric partition is a descriptive computation allocation, not final-logit causal mediation or standalone extraction.'};(P/'QUERY_SOURCE_CUE_PORTS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(corners=x),P/'QUERY_SOURCE_CUE_PORTS_V1_ARTIFACT.pt');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
