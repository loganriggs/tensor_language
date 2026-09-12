"""Exact tests of quadratic readout sufficiency given linear features and norm.

No claim that the norm itself has a closed update rule.
"""
import torch


def dense_certificate(s,b):
    d=s.shape[0];q=torch.eye(d,dtype=s.dtype,device=s.device)-b@b.T
    cross=q@s@b
    if b.shape[1]==d:return dict(cross=cross,anisotropy=q@s@q,beta=s.new_zeros(()))
    beta=torch.trace(q@s)/(d-b.shape[1])
    return dict(cross=cross,anisotropy=q@s@q-beta*q,beta=beta)


def controls():
    torch.manual_seed(6132701);d=6;b=torch.eye(d,dtype=torch.float64)[:,:2]
    a=torch.tensor([[2.,.7],[.7,-1.]],dtype=torch.float64);q=torch.eye(d,dtype=torch.float64)-b@b.T
    s=b@a@b.T+3*q;c=dense_certificate(s,b);x=torch.randn(40,d,dtype=torch.float64)
    z=x@b;norm=x.square().sum(-1);eps=torch.finfo(torch.float32).eps
    direct=torch.einsum('nd,de,ne->n',x,s,x)/(norm/d+eps)
    compiled=(torch.einsum('nr,rs,ns->n',z,a,z)+3*(norm-z.square().sum(-1)))/(norm/d+eps)
    replay=float((direct-compiled).norm()/direct.norm());assert replay<1e-12
    assert c['cross'].norm()<1e-12 and c['anisotropy'].norm()<1e-12
    bad=s.clone();bad[2,2]+=1;assert dense_certificate(bad,b)['anisotropy'].norm()>.5
    # Same linear feature and norm; the visible residual output closes, but its
    # new full norm does not. B(x)= (x0^2,x0^2)/rho(x)^2.
    pair=torch.tensor([[1.,1.],[1.,-1.]],dtype=torch.float64)
    update=(pair[:,0].square()/(pair.square().mean(-1)+eps))[:,None].expand(-1,2)
    y=pair+update;assert y[0,0]==y[1,0]
    norm_gap=float(y[0].square().sum()-y[1].square().sum());assert norm_gap>3.9
    return dict(positive_replay_error=replay,anisotropic_complement_rejected=True,closed_reader_but_open_next_norm_gap=norm_gap)


if __name__=='__main__':
    import json
    from pathlib import Path
    result=controls();Path(__file__).with_name('NORM_AWARE_FEATURE_FIBERS_V1_CONTROL.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
