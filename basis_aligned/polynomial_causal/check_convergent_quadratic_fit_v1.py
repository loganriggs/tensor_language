"""Planted one-product recovery, unfinished-chunk and checkpoint-resume control."""
import hashlib,json,io,time
from pathlib import Path
import torch
from structured_quadratic_models_v1 import QuadraticModel,QuadraticObjective
from convergent_quadratic_fit_v1 import advance
from joint_quadratic_fit_v1 import product_cross

def main():
    tic=time.perf_counter();torch.manual_seed(9116021);torch.set_num_threads(2);dt=torch.float64
    l=torch.randn(1,5,dtype=dt);r=torch.randn(1,5,dtype=dt);d=torch.randn(3,1,dtype=dt);m=torch.eye(3,dtype=dt)
    total=((d.T@m@d)*product_cross(l,r,l,r)).sum();objective=QuadraticObjective(m,total,l=l,r=r,d=d)
    model=QuadraticModel('product',5,products=1)
    first=advance(model,objective,seconds=0,adam_steps=30,diagnostic_every=10)
    assert not first['converged'] and first['terminal_reason']=='chunk_time_limit'
    buffer=io.BytesIO();torch.save(first,buffer);buffer.seek(0);restored=torch.load(buffer,weights_only=False)
    resumed=QuadraticModel('product',5,products=1);resumed.load_state_dict(restored['model'])
    final=advance(resumed,objective,restored,seconds=10,adam_steps=30,diagnostic_every=10)
    assert final['converged'] and final['best']['diagnostics']['squared_relative_error']<=1e-8
    result=dict(unfinished_chunk_not_converged=True,checkpoint_resume_converged=final['converged'],
        best=final['best']['diagnostics'],last=final['history'][-1],cpu_seconds=time.perf_counter()-tic,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    target=Path(__file__).with_name('CONVERGENT_QUADRATIC_FIT_V1_CONTROL.json')
    with target.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
