"""Native head9 cache control and complete value-sector geometry; no selection/fitting."""
import json,torch
import torch.nn.functional as F
from pathlib import Path
from scalar_value_sectors_v1 import head_scalar_sectors
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.set_num_threads(2);out=P/'SCALAR_VALUE_SECTORS_V1_CONTROL.json';assert not out.exists()
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];fields=torch.zeros(2,48,22,2,dtype=torch.float64)
 for i,r in enumerate(rows):
  n=len(r['ids']);tokens=torch.tensor([r['ids']])
  for arm in range(2):fields[arm,i,:n]=head_scalar_sectors(F.rms_norm(cache['r9'][arm,i,:n],(1152,))[None],tokens,p,1)[0]
 replay=float((fields.sum(-1)-cache['a9']).norm()/cache['a9'].norm());cells=[]
 for family in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==family]
  for kind,delta in [('cue',fields[0,ix][::2]-fields[0,ix][1::2]),('after8removal',fields[1,ix]-fields[0,ix])]:
   current=delta[...,0];first=delta[...,1];full=current+first
   cells.append(dict(family=family,change=kind,current_to_full_norm=float(current.norm()/full.norm()),first_to_full_norm=float(first.norm()/full.norm()),current_first_cosine=float(F.cosine_similarity(current.flatten(),first.flatten(),dim=0))))
 result={'pred_a':replay<=1e-5,'native_replay':replay,'cells':cells,'scope':'Exact source-sector split of fixed head9 component, shared jointQK and writer. Norms/cosines are descriptive, not end-task selective-circuit evidence.'}
 out.write_text(json.dumps(result,indent=2)+'\n');torch.save(dict(fields=fields),P/'SCALAR_VALUE_SECTORS_V1_ARTIFACT.pt');print(json.dumps(result))
if __name__=='__main__':main()
