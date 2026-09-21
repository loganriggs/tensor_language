"""Check shared quadratic cumulants against exact degree-eight quadrature."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from shared_gaussian_moments import gram,native_cross,PARTITIONS
from noncentral_gaussian_cp import project_shifted
from quartic_cp import directional
from sparse_quartic_bank import support
P=Path(__file__).resolve().parent

def controls():
    torch.set_num_threads(2);dtype=torch.float64;rows=[]
    nodes,weights=np.polynomial.hermite.hermgauss(5);idx=torch.tensor(list(itertools.product(range(5),repeat=3)))
    z=torch.tensor(nodes*2**.5,dtype=dtype)[idx];w=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype)[idx].prod(1)
    for seed in range(5):
        torch.manual_seed(11800+seed);U=torch.randn(4,2,3,dtype=dtype);V=torch.randn_like(U)
        if seed==1:U[1]=U[0]
        if seed==2:V[1]=V[0]
        if seed==3:V=U.clone()
        if seed==4:U[1]=U[0];V[1]=-V[0]
        bu=torch.randn(4,2,dtype=dtype);bv=torch.randn_like(bu)
        for shifted in [False,True]:
            params=[a.clone().requires_grad_() for a in [U,V,bu if shifted else torch.zeros_like(bu),bv if shifted else torch.zeros_like(bv)]]
            u,v,b,c=params;pairs=support(4,7,11900);g=gram(u,v,pairs,b,c,chunk=3)
            q=(((z@u.flatten(0,1).T).reshape(len(z),4,2)+b)*((z@v.flatten(0,1).T).reshape(len(z),4,2)+c)).sum(-1)
            phi=q[:,pairs[0]]*q[:,pairs[1]];ref=phi.T@(w[:,None]*phi)
            teacher=[torch.randn(*shape,dtype=dtype) for shape in [(2,4),(4,3),(4,3),(3,4),(4,3),(4,3)]]
            location=torch.randn(3,dtype=dtype) if shifted else torch.zeros(3,dtype=dtype)
            projection=project_shifted(teacher,location)
            X=native_cross(teacher,location,projection,u,v,pairs,b,c,chunk=2)
            y=directional(*teacher,[z+location]*4);xref=y.T@(w[:,None]*phi)
            probe=torch.randn_like(g);xprobe=torch.randn_like(X);ga=torch.autograd.grad((g*probe).sum()+(X*xprobe).sum(),params,retain_graph=True);gb=torch.autograd.grad((ref*probe).sum()+(xref*xprobe).sum(),params)
            rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-15)).detach())
            cross_error=rel(X,xref);error=rel(g,ref);gradient=rel(torch.cat([a.flatten() for a in ga]),torch.cat([a.flatten() for a in gb]))
            assert error<1e-10 and gradient<1e-10 and cross_error<1e-10,(seed,shifted,error,gradient)
            rows.append(dict(seed=seed,shifted=shifted,gram_error=error,cross_error=cross_error,gradient_error=gradient))
    assert len(PARTITIONS)==15
    return rows
if __name__=='__main__':
    rows=controls();(P/'SHARED_GAUSSIAN_MOMENTS_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),max_gram_error=max(r['gram_error'] for r in rows),max_gradient_error=max(r['gradient_error'] for r in rows))))
