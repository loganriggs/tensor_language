import json,time
from pathlib import Path
import torch
from exact_cp_fit import fit
P=Path(__file__).resolve().parent

def main():
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);source=json.loads((P/'EXACT_CP_RESIDUAL_CONTROLS_V1.json').read_text());tc=torch.tensor(source['teacher']['C']);teacher=[torch.tensor(v) for v in source['teacher']['factors']];out=P/'EXACT_CP_SCALE_CONTROL_V1.json';assert not out.exists();rows=[];start=time.perf_counter()
 for opt in ['adam','muon']:
  for seed in [0,1]:
   torch.manual_seed(seed);initial=[torch.randn_like(v) for v in teacher];initial=[v/v.norm(dim=1,keepdim=True) for v in initial];row,_,_=fit(tc,teacher,2,opt,seed,1500,initial=initial);row.update(optimizer=opt,seed=seed,method='joint_unit_raw1500');rows.append(row);print(opt,seed,row['relative_error'],flush=True);out.write_text(json.dumps(dict(records=rows,seconds=time.perf_counter()-start,scope='Same random initial directions as prior joint control, changed raw parameter norms only. One fixed planted teacher; not native generalization.'),indent=2)+'\n')
if __name__=='__main__':main()
