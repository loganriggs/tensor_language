from pathlib import Path
import json,time,torch
import torch.nn.functional as F
from regional_even_query_source_v1 import scalar
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);tic=time.perf_counter();cache=torch.load(P/'SELECTIVE_INTERACTION_PORTS_V1_ARTIFACT.pt',weights_only=True);rows=json.loads((P/'SCALAR_CUE_CHANNEL_SHIFT_V1_ROWS.json').read_text())['rows'];p=torch.load(P/'extracted_circuits/regional_even_key_producers_8_2_9_8_v1/program.pt',weights_only=True);native=F.rms_norm(cache['raw9'][0],(1152,));corners=[];rel=lambda a,b:float((a-b).norm()/b.norm().clamp_min(1e-30))
 for i in range(0,72,2):
  n=len(rows[i]['ids']);u=native[i:i+1,:n];a=native[i+1:i+2,:n];ids=torch.tensor([rows[i]['ids']]);values=[]
  for k,v in [(u,u),(a,u),(u,a),(a,a)]:values.append(scalar(a,k,ids,p,1,value_source=v)[0,-1])
  corners.append(torch.stack(values))
 x=torch.stack(corners);prior=torch.load(P/'QUERY_SOURCE_CUE_PORTS_V1_ARTIFACT.pt',weights_only=True)['corners'];anchor=rel(x[:,[0,3]],prior[:,[1,3]]);target=x[:,0]-x[:,3];key=((x[:,0]-x[:,1])+(x[:,2]-x[:,3]))/2;value=((x[:,0]-x[:,2])+(x[:,1]-x[:,3]))/2;interaction=x[:,0]-x[:,1]-x[:,2]+x[:,3];records=[]
 for family in range(3):
  ix=[i//2 for i in range(0,72,2) if rows[i]['family']==family];d=target[ix];k=key[ix];v=value[ix];records.append(dict(family=family,name=rows[ix[0]*2]['family_name'],key_aligned_fraction=float((k@d)/d.square().sum()),value_aligned_fraction=float((v@d)/d.square().sum()),key_only_error=rel(x[ix,2]-x[ix,3],d),value_only_error=rel(x[ix,1]-x[ix,3],d),interaction_norm_over_target=float(interaction[ix].norm()/d.norm())))
 accounting=rel(key+value,target);A=anchor<=1e-10 and accounting<=1e-10;result={'pred_a':A,'pred_b':A and all(r['key_only_error']<=.2 for r in records),'pred_c':A and all(r['value_only_error']<=.2 for r in records),'tied_corner_anchor':anchor,'accounting_error':accounting,'records':records,'seconds':time.perf_counter()-tic,'scope':'Same36paired nativecue contexts, query fixedAmerican. TwoQKsourcekeys tied asone factor, value another. SourceKV contrast from priorfactorial is target, not fullnativecontrast. Local computational allocation, not signed logit mediation.'};(P/'SOURCE_KEY_VALUE_CUE_PORTS_V1_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(corners=x),P/'SOURCE_KEY_VALUE_CUE_PORTS_V1_ARTIFACT.pt');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
