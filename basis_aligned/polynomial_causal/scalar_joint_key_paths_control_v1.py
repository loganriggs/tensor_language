"""Native cache control of frozen weight-only joint-key product bank.
A: sum10paths and2value sectors replays pristine/changed scalar9<=1e-5relative.
Native FP32 rounding is retained in reference and key normalizers.
"""
import json,time,torch
import torch.nn.functional as F
from pathlib import Path
from scalar_joint_key_paths_v1 import paths,PAIRS
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(2);tic=time.perf_counter();out=P/'SCALAR_JOINT_KEY_PATHS_V1_CONTROL.json';assert not out.exists()
 p=torch.load(P/'SCALAR_PRODUCERS_COMPILE_V1_PROGRAM.pt',weights_only=True,map_location='cpu');bands=torch.load(P/'SCALAR_JOINT_KEY_BANK_V1_ARTIFACT.pt',weights_only=True,map_location='cpu')['bands'];c=torch.load(P/'SCALAR_PRODUCER_MLP_BRIDGE_CACHE_V1_ARTIFACT.pt',weights_only=True,map_location='cpu');rows=json.loads((P/'STRUCTURED_PRODUCER_CONFIRMATION_V1_ROWS.json').read_text())['rows']
 v=torch.zeros(2,48,22,10,2,dtype=torch.float64)
 for i,row in enumerate(rows):
  n=len(row['ids']);x=F.rms_norm(c['r9'][:,i,:n],(1152,),eps=torch.finfo(torch.float32).eps);tokens=torch.tensor([row['ids']]).expand(2,-1)
  v[:,i,:n]=paths(x,tokens,p,1,bands[1])
 errors=[float((v[a].sum((-1,-2))-c['a9'][a]).norm()/c['a9'][a].norm()) for a in range(2)]
 delta=v[1]-v[0];actual=c['a9'][1]-c['a9'][0];err=float((delta.sum((-1,-2))-actual).norm()/actual.norm())
 cells=[]
 for fam in range(2):
  ix=[i for i,r in enumerate(rows) if r['family']==fam];pieces=delta[ix].sum(-1);full=actual[ix]
  cells.append(dict(family=fam,individual_path_change_norm_fractions=[float(pieces[:,:,j].norm()/full.norm()) for j in range(10)],sum_path_norms_over_full=sum(float(pieces[:,:,j].norm()/full.norm()) for j in range(10))))
 result=dict(pred_a=max(errors+[err])<=1e-5,native_replay=errors,scalar_change_replay=err,cells=cells,pairs=PAIRS,seconds=time.perf_counter()-tic,scope='All path terms kept. Intermediate native9 control on reused contexts, no path selection/refit or semantic/behavioral claim. Native8 control still required in GPU screen; spectral gaps are not identification.')
 torch.save(dict(paths=v),P/'SCALAR_JOINT_KEY_PATHS_V1_CONTROL_ARTIFACT.pt');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert result['pred_a']
if __name__=='__main__':main()
