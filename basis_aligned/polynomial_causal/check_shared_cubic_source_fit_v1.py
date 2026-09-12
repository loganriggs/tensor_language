"""Cold-start recovery of a shared source cubic under private query maps."""
from pathlib import Path
import json,time
import torch
from shared_cubic_source_projection_v1 import capture
from shared_cubic_source_fit_v1 import fit
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73320);dt=torch.float64
    source=torch.randn(4,dtype=dt);source/=source.norm()
    q1=torch.randn(2,2,3,dtype=dt);q2=torch.randn_like(q1)
    k1=torch.randn(2,2,1,dtype=dt)*source;k2=torch.randn(2,2,1,dtype=dt)*source
    v=torch.randn(2,2,1,dtype=dt)*source;o=torch.randn(3,2,2,dtype=dt);weights=(q1,k1,q2,k2,v,o)
    planted=source.expand(1,3,-1).clone();full=float(capture(planted,*weights));reports=[]
    for seed in (73321,73322,73323,73324):
        torch.manual_seed(seed);start=torch.randn(1,3,4,dtype=dt);tic=time.perf_counter()
        atoms,history,reason=fit(start,weights,full,max_steps=512,max_seconds=30)
        loss=1-float(capture(atoms,*weights))/full;reports.append(dict(seed=seed,relative_squared_error=loss,gradient=history[-1]['projected_gradient_norm'],reason=reason,steps=len(history),seconds=time.perf_counter()-tic))
    result=dict(reports=reports,all_recovered=all(abs(z['relative_squared_error'])<1e-8 for z in reports),all_stationary=all(z['reason']=='stationary' for z in reports))
    out=P/'SHARED_CUBIC_SOURCE_FIT_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
