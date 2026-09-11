"""CPU-only same-shape manifold kernel timing; no native GPU fit."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_info,threadpool_limits
from orthogonal_multioutput_pymanopt_v2 import manifold_for

def main():
    p=Path(__file__).resolve().parent;r=json.loads((p/'BLOCK_TRUST_REGION_V1_RESULT.json').read_text());cache=r['cache']
    assert hashlib.sha256(Path(cache['path']).read_bytes()).hexdigest()==cache['sha256']
    saved=torch.load(cache['path'],weights_only=True,map_location='cpu');bank,packed=saved['bank'],saved['packed']
    man=manifold_for(bank,packed);point=[bank.numpy(),packed.numpy()];gen=np.random.default_rng(1559)
    grad=[gen.standard_normal(x.shape) for x in point];hess=[gen.standard_normal(x.shape) for x in point]
    direction=man.projection(point,[gen.standard_normal(x.shape) for x in point]);direction/=man.norm(point,direction)
    def operation():
        a=man.euclidean_to_riemannian_gradient(point,grad)
        b=man.euclidean_to_riemannian_hessian(point,grad,hess,direction)
        return a,b
    original=threadpool_info();rows=[];reference=None;error=0.
    # Reverse confirmation pass prevents a simple warm-cache order explanation.
    for threads in [16,2,1,1,2,16]:
        with threadpool_limits(limits=threads,user_api='blas'):
            for _ in range(3):out=operation()
            start=time.perf_counter();cpu=time.process_time()
            for _ in range(20):out=operation()
            elapsed=time.perf_counter()-start;process=time.process_time()-cpu
            if reference is None:reference=out
            for a,b in zip(out,reference):
                error=max(error,float(np.sqrt(sum(np.sum((x-y)**2) for x,y in zip(a,b))/sum(np.sum(y*y) for y in b))))
            rows.append(dict(threads=threads,wall_seconds=elapsed,cpu_seconds=process,repeats=20,
                runtime_threads=[x['num_threads'] for x in threadpool_info() if x['user_api']=='blas']))
    mean=lambda t,k:sum(x[k] for x in rows if x['threads']==t)/2
    wall_ratio=mean(2,'wall_seconds')/mean(16,'wall_seconds');cpu_ratio=mean(2,'cpu_seconds')/mean(16,'cpu_seconds')
    result=dict(predictions={'pred_a_same_operation':error<=1e-10,'pred_b_wall_speedup':wall_ratio<=.5,'pred_c_cpu_reduction':cpu_ratio<=.5},
        maximum_relative_output_error=error,two_vs_sixteen_wall_ratio=wall_ratio,two_vs_sixteen_cpu_ratio=cpu_ratio,
        wall_speedup=1/wall_ratio,rows=rows,original_pools=original,
        scope='Matched-shape host manifold kernels only; both caps forced in this process, CUDA hidden. Not end-to-end native solver speedup or convergence proof.')
    with (p/'BLOCK_HOST_BLAS_V1_AUDIT.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['predictions']['pred_a_same_operation']

if __name__=='__main__':main()
