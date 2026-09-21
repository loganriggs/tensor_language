import json
from pathlib import Path
import torch
from gaussian_bank_metric import gram_by_degree
from projected_quartic_gaussian import functional_cross,projected_second
from empirical_quartic_dictionary import features
from dag_square_optimizer import rule


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    x,w=rule(5);x=x[:,:4];rows=[]
    for seed in range(5):
        torch.manual_seed(4400+seed);rand=lambda *s:torch.randn(*s)
        teacher=[rand(*s) for s in [(3,5),(5,6),(5,6),(6,7),(7,4),(7,4)]]
        u,v=rand(3,2,4),rand(3,2,4)
        if seed==1:u[1]=u[0]
        if seed==2:v[2]=v[0]+v[1];u[2]=u[0]
        if seed==3:v=u.clone()
        if seed==4:u[2]=u[0];v[2]=-v[0]
        t=rand(4,4)/2+torch.eye(4);tt=[*teacher[:-2],teacher[-2]@t,teacher[-1]@t];a,b=u@t,v@t
        G,meta=gram_by_degree(a,b);got,parts=functional_cross(tt,a,b,meta,pair_batch=4,channel_batch=2)
        C,l,r,D,A,B=teacher;z=x@t.T;h=((z@A.T)*(z@B.T))@D.T;y=((h@l.T)*(h@r.T))@C.T;phi=features(z,u,v)
        reference=y.T@(w[:,None]*phi);error=float((got-reference).norm()/reference.norm())
        # Independent normal equations: exact moments versus quadrature regression.
        gram=sum(G.values());s=gram.diag().sqrt();system=gram/s[:,None]/s[None,:]+1e-6*torch.eye(len(s))
        cw=torch.linalg.solve(system,(got/s).T).T/s
        cq=torch.linalg.solve(system,(reference/s).T).T/s
        solve_error=float((cw-cq).norm()/cq.norm())
        batch_error=float((projected_second(tt,meta['basis'],1)-projected_second(tt,meta['basis'],3)).norm()/parts['degree2'].norm())
        assert max(error,solve_error,batch_error)<1e-9
        rows.append(dict(seed=seed,cross_error=error,writer_error=solve_error,batch_error=batch_error))
    # Wide root count and nondivisible native-channel batches without native weights.
    u,v=rand(32,4,4),rand(32,4,4);G,meta=gram_by_degree(u,v);got,_=functional_cross(teacher,u,v,meta,8,2)
    assert got.shape==(3,528) and sum(G.values()).shape==(528,528)
    result=dict(records=rows,wide_shape=list(got.shape),scope='Exact Gaussian teacher-to-bank cross and readout controls against independent degree8 quadrature; native-format teacher, covariance transforms and partial batches.')
    Path(__file__).with_name('PROJECTED_QUARTIC_GAUSSIAN_CONTROLS_V1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
