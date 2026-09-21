import json
from pathlib import Path
import torch
from quartic_joint_readout import solve,solve_rank
from empirical_quartic_dictionary import features
from quartic_bank_derivative import feature_jacobian


def main():
    torch.set_num_threads(2);torch.set_default_dtype(torch.float64);torch.manual_seed(5600);rows=[]
    for family in ['independent','shared_input','shared_output','squares','cancellation']:
        u,v=torch.randn(3,2,4),torch.randn(3,2,4);c=torch.randn(5,6)
        if family=='shared_input':u[1]=u[0]
        if family=='shared_output':c=torch.randn(5,1)@torch.randn(1,6)
        if family=='squares':v=u.clone()
        if family=='cancellation':u[2]=u[0];v[2]=-v[0]
        x=torch.randn(61,4);phi=features(x,u,v);y=phi@c.T+.1*torch.randn(61,5);jac=[feature_jacobian(u,v,z) for z in x[:5]];target=[c@j+.1*torch.randn(5,4) for j in jac];dg=sum(j@j.T for j in jac);dc=sum(t@j.T for t,j in zip(target,jac));de=sum(t.square().sum() for t in target)
        full,info=solve(phi,y,dg,dc,de,1.);s=info['scales'];ve=y.square().sum()
        design=torch.cat([phi/s/ve.sqrt(),*(j.T/s/de.sqrt() for j in jac),info['ridge'].sqrt()*torch.eye(6)]);outputs=torch.cat([y/ve.sqrt(),*(t.T/de.sqrt() for t in target),torch.zeros(6,5)])
        q=torch.linalg.qr(design,mode='reduced').Q;optimal_prediction=q@(q.T@outputs);uu,ss,vh=torch.linalg.svd(optimal_prediction,full_matrices=False)
        max_error=0.;cert=0.
        for rank in [1,2,5]:
            writer,basis,check=solve_rank(phi,y,dg,dc,de,1.,rank)
            got=design@(writer*s).T;reference=(uu[:,:rank]*ss[:rank])@vh[:rank];error=float((got-reference).norm()/reference.norm());max_error=max(max_error,error);cert=max(cert,check['spectral_objective_relative_error']);assert max(error,cert)<1e-8
            if rank==5:assert float((phi@(writer-full).T).norm()/y.norm())<1e-8
        rows.append(dict(family=family,stacked_low_rank_prediction_error=max_error,spectral_certificate_error=cert))
    Path(__file__).with_name('QUARTIC_RANK_READOUT_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
