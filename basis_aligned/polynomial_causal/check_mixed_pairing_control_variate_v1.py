"""Exhaustive coefficient and on-manifold gradient checks, no stochastic tolerance."""
import itertools,json,math
from pathlib import Path
import torch
from mixed_pairing_control_variate_v1 import pairing_values,sampled_remainder
from mixed_repeated_contraction_v1 import contract
from graded_source_projection_v1 import graded_norms,retained_grades
P=Path(__file__).resolve().parent
def main():
    torch.set_num_threads(2);torch.manual_seed(73230);dtype=torch.float64
    l=torch.randn(3,2,dtype=dtype);r=torch.randn_like(l);d=torch.randn(2,3,dtype=dtype)
    a=torch.randn(3,2,2,dtype=dtype);a=(a+a.transpose(-1,-2))/2
    w=torch.randn(2,3,dtype=dtype);wg=w@w.T
    m=torch.einsum('ah,hi,hj->aij',d,l,r);m=(m+m.transpose(-1,-2))/2
    h=m.flatten(1)@m.flatten(1).T;lam,v=torch.linalg.eigh(h);root=(v*lam.sqrt())@v.T
    theta=torch.tensor(.37,dtype=dtype,requires_grad=True);p=torch.stack([theta.cos(),theta.sin()])[:,None]
    dd=root@p@(p.T@torch.linalg.solve(root,d));full=graded_norms(a,h,wg);ret=retained_grades(a,root,p,wg)
    reports=[]
    for k in range(1,5):
        ids=torch.tensor(list(itertools.product(range(2),repeat=4+k)));vectors=torch.eye(2,dtype=dtype)[ids]
        b=vectors[:,:4-k];x=vectors[:,4-k:]
        ref=pairing_values(b,x,l,r,d,a,k);approx=pairing_values(b,x,l,r,dd,a,k);e=approx-ref
        oracle=contract(b,x,l,r,dd,a,k)-contract(b,x,l,r,d,a,k)
        norm=torch.einsum('ni,ij,nj->',oracle,wg,oracle)
        analytic=math.comb(4,k)*(full[k]-ret[k])/e.shape[-1]
        cv=sampled_remainder(e,wg).sum()+analytic
        pairnorm=torch.einsum('ni,ij,nj->',e[:,:,0],wg,e[:,:,0])
        g1=torch.autograd.grad(norm,theta,retain_graph=True)[0];g2=torch.autograd.grad(cv,theta,retain_graph=True)[0]
        reports.append(dict(grade=k,value_error=float(((norm-cv).abs()/norm.abs()).detach()),gradient_error=float(((g1-g2).abs()/g1.abs().clamp_min(1e-12)).detach()),pairing_mean_error=float(((e.mean(-1)-oracle).norm()/oracle.norm()).detach()),paired_norm_error=float(((pairnorm-analytic*e.shape[-1]).abs()/pairnorm).detach())))
    assert max(v for z in reports for key,v in z.items() if key!='grade')<1e-9
    out=P/'MIXED_PAIRING_CONTROL_VARIATE_V1_CONTROL.json';assert not out.exists();out.write_text(json.dumps(reports,indent=2)+'\n');print(reports)
if __name__=='__main__':main()
