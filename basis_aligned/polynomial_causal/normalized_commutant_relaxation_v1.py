"""Dense control of an exact generalized relaxation, not a native-size solver."""
import json
from pathlib import Path
import torch
from fullu_input_blocks_v1 import sandwich,commutator,partition_metrics
from toy_consumer_commutant_blocks import planted_family,dense_null_family


def symmetric_basis(d):
    result=[]
    for i in range(d):
        for j in range(i,d):
            a=torch.zeros(d,d)
            if i==j:a[i,j]=1
            else:a[i,j]=a[j,i]=2**-.5
            result.append(a)
    return torch.stack(result)


def dense_relaxation(l,r,g):
    """Small-dimensional full generalized eigensolve with identity removed."""
    d=l.shape[1];basis=symmetric_basis(d)
    k=sandwich(l,r,g,torch.eye(d))
    actions=torch.stack([commutator(l,r,g,b,k) for b in basis])
    mass=torch.stack([k@b+b@k for b in basis])
    operator=torch.einsum('aij,bij->ab',basis,actions)
    m=torch.einsum('aij,bij->ab',basis,mass)
    me,mv=torch.linalg.eigh(m);assert float(me.min())>0
    sqrt=(mv*me.sqrt())@mv.T;inv=(mv*me.rsqrt())@mv.T
    ident=torch.einsum('aij,ij->a',basis,torch.eye(d))
    trivial=sqrt@ident;trivial/=trivial.norm()
    # QR supplies an orthogonal complement to the one trivial vector.
    complement=torch.linalg.qr(trivial[:,None],mode='complete').Q[:,1:]
    reduced=complement.T@inv@operator@inv@complement
    values,vectors=torch.linalg.eigh((reduced+reduced.T)/2)
    v=inv@complement@vectors[:,0]
    x=torch.einsum('a,aij->ij',v,basis)
    residual=float((operator@v-values[0]*m@v).norm()/(operator.norm()*v.norm()))
    _,q=torch.linalg.eigh(x);p=q[:,:d//2]@q[:,:d//2].T
    return dict(eigenvalues=values.tolist(),numerical_relaxation_minimum=float(values[0]),
                eigenpair_relative_residual=residual,
                rounded=partition_metrics(l,r,g,p,k)),p


def control():
    torch.set_default_dtype(torch.float64);d=8
    l=torch.eye(d).repeat_interleave(d,0);r=torch.eye(d).repeat(d,1)
    results=[]
    for kind in ('planted','dense'):
        forms=planted_family(seed=120432,block_sizes=(4,4),consumers=10)[0] if kind=='planted' else dense_null_family(seed=120432,dimension=d,consumers=10)
        w=forms.reshape(10,-1);g=w.T@w
        result,p=dense_relaxation(l,r,g)
        k=sandwich(l,r,g,torch.eye(d));a=torch.trace(p@k);total=torch.trace(k)
        x=p-a/total*torch.eye(d)
        rayleigh=(x*commutator(l,r,g,x,k)).sum()/(x*(k@x+x@k)).sum()
        result['rayleigh_cut_identity_error']=abs(float(rayleigh)-result['rounded']['normalized_cut'])
        result['kind']=kind;results.append(result)
    old=json.loads(Path(__file__).with_name('FULLU_BLOCK_OPTIMIZER_V1_CONTROL.json').read_text())['reports']
    dense_min=min(x['normalized_cut'] for x in old if x['kind']=='dense')
    passed=results[0]['rounded']['normalized_cut']<1e-10 and results[1]['numerical_relaxation_minimum']<=dense_min+1e-10 and all(x['rayleigh_cut_identity_error']<1e-12 and x['eigenpair_relative_residual']<1e-12 for x in results)
    return dict(results=results,previous_dense_local_minimum=dense_min,passed=passed,
                limitation='Dense small control; native Krylov Ritz values are not certified lower bounds.')


if __name__=='__main__':
    result=control();Path(__file__).with_name('NORMALIZED_COMMUTANT_RELAXATION_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));assert result['passed']
