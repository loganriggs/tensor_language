"""Frozen-bank robustness to near/far weight-only position construction.
A within-band rotation function error<=1e-10. B top-top native9 field<=.1error
bothpositionhalves/bothfamilies. C top64overlap>=.9 bothheads/bothhalves.
No change to queued candidates, no text fitting. See board preregistration.
"""
import json,time,torch
import torch.nn.functional as F
from pathlib import Path
from joint_qk_source_influence_v1 import influence
from folded_normalized_router_v1 import rotary
from scalar_joint_key_paths_v1 import paths
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'SCALAR_JOINT_KEY_POSITION_AUDIT_V1_RESULT.json';assert not out.exists()
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');frozen=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'];banks=[];geometry=[]
 for h in range(2):
  q1,q2,k1,k2=[p[n][h].double() for n in ('q1','q2','k1','k2')];basis=torch.linalg.qr(torch.cat((k1,k2)).T,mode='reduced').Q;H=[]
  for part in range(2):
   terms=[]
   for pos in range(part*16,(part+1)*16):
    rot=rotary(32,128).T@rotary(pos,128);terms.append(influence(q1,q2,rot@k1@basis,rot@k2@basis))
   e,u=torch.linalg.eigh(torch.stack(terms).mean(0));b=(basis@u.flip(1)).reshape(1152,4,64).permute(1,0,2).contiguous();H.append(b)
   geometry.append(dict(head=h,position_half=part,band_overlaps=[float((b[j].T@frozen[h,j]).square().sum()/64) for j in range(4)]))
  banks.append(torch.stack(H))
 banks=torch.stack(banks);cache=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');reference=torch.load(P/'SCALAR_JOINT_KEY_PATHS_V1_CONTROL_ARTIFACT.pt',weights_only=True,map_location='cpu')['paths'][0]
 rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows'];v=torch.zeros(2,48,22,10,2,dtype=torch.float64);rotation_error=0.
 generator=torch.Generator().manual_seed(191021)
 for i,row in enumerate(rows):
  n=len(row['ids']);x=F.rms_norm(cache['r9'][0,i,:n],(1152,),eps=torch.finfo(torch.float32).eps)[None];tokens=torch.tensor([row['ids']])
  for part in range(2):v[part,i,:n]=paths(x,tokens,p,1,banks[1,part])[0]
  if i==0:
   rotated=torch.stack([b@torch.linalg.qr(torch.randn(64,64,dtype=torch.float64,generator=generator)).Q for b in frozen[1]])
   old=paths(x,tokens,p,1,frozen[1]);new=paths(x,tokens,p,1,rotated);rotation_error=float((new-old).norm()/old.norm())
 cells=[]
 for part in range(2):
  for family in range(2):
   ix=[i for i,r in enumerate(rows) if r['family']==family]
   field_errors=[float((v[part,ix,:,j].sum(-1)-reference[ix,:,j].sum(-1)).norm()/reference[ix,:,j].sum(-1).norm()) for j in range(10)]
   cells.append(dict(position_half=part,family=family,path_field_errors=field_errors))
 result=dict(pred_a=rotation_error<=1e-10,pred_b=rotation_error<=1e-10 and all(c['path_field_errors'][0]<=.1 for c in cells),pred_c=rotation_error<=1e-10 and all(g['band_overlaps'][0]>=.9 for g in geometry),within_band_rotation_error=rotation_error,geometry=geometry,cells=cells,seconds=time.perf_counter()-tic,scope='Position-weight construction sensitivity of frozen complete joint-key bands. Native9 field on reused rows, not effect, OOD, or task selectivity. All queued candidate identities remain unchanged.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
