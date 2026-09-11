"""Same Stiefel geometry with association avoiding an n-by-n intermediate.

Pymanopt2.2.1 weingarten forms (V X^T)N. Use V(X^T N), identically in real
arithmetic; QR retraction, projection and the optimizer remain inherited.
"""
import json,time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from pymanopt.manifolds import Stiefel,Product,Oblique
from pymanopt.tools.multi import multisym,multitransp


class ThinStiefel(Stiefel):
    def weingarten(self,point,tangent_vector,normal_vector):
        return -(tangent_vector@(multitransp(point)@normal_vector))-point@multisym(multitransp(tangent_vector)@normal_vector)


def manifold_for(bank,packed):
    groups,dim,rank=bank.shape
    return Product([ThinStiefel(dim,rank,k=groups),Oblique(*packed.shape)])


def control():
    p=Path(__file__).resolve().parent;r=json.loads((p/'BLOCK_TRUST_REGION_V1_RESULT.json').read_text())
    s=torch.load(r['cache']['path'],weights_only=True,map_location='cpu');b,c=s['bank'],s['packed'];point=[b.numpy(),c.numpy()]
    reference=Product([Stiefel(1152,16,k=16),Oblique(136,64)]);thin=manifold_for(b,c)
    gen=np.random.default_rng(1571);g=[gen.standard_normal(x.shape) for x in point];h=[gen.standard_normal(x.shape) for x in point]
    v=reference.projection(point,[gen.standard_normal(x.shape) for x in point]);v/=reference.norm(point,v)
    def apply(man):return man.euclidean_to_riemannian_hessian(point,g,h,v)
    rows=[];errors=[]
    for threads in [16,2]:
        with threadpool_limits(limits=threads,user_api='blas'):
            ref=apply(reference);new=apply(thin)
            errors.append(float(np.sqrt(sum(np.sum((a-b)**2) for a,b in zip(ref,new))/sum(np.sum(a*a) for a in ref))))
            for man,name in [(reference,'reference'),(thin,'thin'),(thin,'thin'),(reference,'reference')]:
                for _ in range(3):apply(man)
                start=time.perf_counter()
                for _ in range(20):apply(man)
                rows.append(dict(threads=threads,method=name,wall_seconds=time.perf_counter()-start,repeats=20))
    mean=lambda threads,method:sum(x['wall_seconds'] for x in rows if x['threads']==threads and x['method']==method)/2
    speeds={str(t):mean(t,'reference')/mean(t,'thin') for t in [16,2]}
    result=dict(predictions={'pred_a_same_hessian':max(errors)<=1e-10,'pred_b_speedup':min(speeds.values())>=5,
        'pred_c_thin_intermediate':True},maximum_relative_error=max(errors),wall_speedups=speeds,rows=rows,
        reference_intermediate_shape=[16,1152,1152],new_intermediate_shape=[16,16,16],
        scope='Same native-shaped Stiefel Hessian conversion, only matrix association changed; no installed-library mutation. Host-kernel benchmark, not native solver speedup.')
    with (p/'THIN_STIEFEL_V1_CONTROL.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result,indent=2));assert result['predictions']['pred_a_same_hessian']

if __name__=='__main__':control()
