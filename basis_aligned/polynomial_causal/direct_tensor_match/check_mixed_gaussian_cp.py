"""Independent dense-coefficient and exact Gaussian-quadrature gradient checks."""
import itertools,json
from pathlib import Path
import numpy as np
import torch
from mixed_gaussian_cp import gram_dynamic,native_mixed_objective
from noncentral_gaussian_cp import gram,project_shifted
from quartic_cp import directional,cp_entries
from quartic_cp_profile import normalize_factors
P=Path(__file__).resolve().parent

def controls():
    torch.set_num_threads(2);dtype=torch.float64;d=3;rows=[]
    nodes,weights=np.polynomial.hermite.hermgauss(5)
    ix=torch.tensor(list(itertools.product(range(5),repeat=d)))
    z=torch.tensor(nodes*2**.5,dtype=dtype)[ix];w=torch.tensor(weights/np.sqrt(np.pi),dtype=dtype)[ix].prod(1)
    indices=torch.tensor(list(itertools.product(range(d),repeat=4)));eye=torch.eye(d,dtype=dtype)
    for seed in range(5):
        torch.manual_seed(11300+seed)
        teacher=[torch.randn(*s,dtype=dtype) for s in [(2,4),(4,3),(4,3),(3,4),(4,d),(4,d)]]
        if seed==1:teacher[4][1]=teacher[4][0]
        if seed==2:teacher[0][1]=teacher[0][0]
        if seed==3:teacher[5]=teacher[4].clone()
        if seed==4:teacher[4][1]=teacher[4][0];teacher[5][1]=-teacher[5][0]
        raw=torch.randn(d,d,dtype=dtype);S=torch.linalg.cholesky(raw@raw.T+.2*eye)
        mu=torch.randn(d,dtype=dtype);x=z@S.T+mu
        transformed=teacher[:4]+[teacher[4]@S,teacher[5]@S];location=torch.linalg.solve(S,mu)
        projection=project_shifted(transformed,location)
        y=directional(*teacher,[x]*4);H=directional(*teacher,[eye[indices[:,s]] for s in range(4)])
        params=[torch.randn(6,d,dtype=dtype,requires_grad=True) for _ in range(4)]
        for weight in [0.,.1,1000.]:
            f=normalize_factors(params);fw=[a@S for a in f];bias=[a@mu for a in f]
            gd,stats=gram_dynamic(fw,bias,fw,bias,True);expanded=gram(fw,bias,fw,bias)
            loss,c=native_mixed_objective(teacher,transformed,location,projection,S,mu,f,weight)
            phi=torch.ones(len(x),6,dtype=dtype)
            for a in f:phi=phi*(x@a.T)
            hc=cp_entries(c,f,indices)
            direct=(weight*((hc-H).square().sum()-H.square().sum())+(w[:,None]*((phi@c.T-y).square()-y.square())).sum())/(1+weight)+1e-6*c.square().sum()
            ga=torch.autograd.grad(loss,params,retain_graph=True);gb=torch.autograd.grad(direct,params)
            rel=lambda a,b:float(((a-b).norm()/b.norm().clamp_min(1e-15)).detach())
            row=dict(seed=seed,coefficient_weight=weight,gram_error=rel(gd,expanded),loss_error=rel(loss,direct),gradient_error=rel(torch.cat([v.flatten() for v in ga]),torch.cat([v.flatten() for v in gb])),**stats)
            assert max(row[k] for k in ['gram_error','loss_error','gradient_error'])<1e-8,row
            rows.append(row)
    return rows
if __name__=='__main__':
    rows=controls();(P/'MIXED_GAUSSIAN_CP_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(dict(cases=len(rows),max_gradient_error=max(r['gradient_error'] for r in rows),recurrence={k:rows[0][k] for k in ['subsets','product_terms']})))
