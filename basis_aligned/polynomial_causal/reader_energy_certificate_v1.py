"""Weight-only certificate for an explicit vocabulary-score spread computation."""
import torch


def reader_gram(U):
    centered=U.double()-U.double().mean(0,keepdim=True)
    return centered.T@centered/len(U)


def certificate(Q,K):
    Q=(Q.double()+Q.double().T)/2;K=(K.double()+K.double().T)/2
    n=len(Q);eye=torch.eye(n,dtype=torch.float64,device=Q.device)
    q0=Q-Q.trace()/n*eye;k0=K-K.trace()/n*eye;den=k0.square().sum()
    if float(den)<=1e-30:
        return dict(identified=False,reason='reader spread has no anisotropic quadratic component')
    slope=(q0*k0).sum()/den;isotropic=(Q.trace()-slope*K.trace())/n
    residual=Q-slope*K-isotropic*eye
    return dict(identified=True,slope=float(slope),isotropic=float(isotropic),relative_anisotropic_residual=float(residual.norm()/q0.norm().clamp_min(1e-30)))


def controls():
    g=torch.Generator().manual_seed(9111218)
    U=torch.randn(19,6,generator=g,dtype=torch.float64);K=reader_gram(U)
    u=torch.randn(11,6,generator=g,dtype=torch.float64);u=u/u.norm(dim=1,keepdim=True)*6**.5
    scores=u@U.T;direct=scores.var(-1,unbiased=False);folded=((u@K)*u).sum(-1)
    gram_error=float((direct-folded).abs().max());assert gram_error<1e-10
    Q=2.3*K-.7*torch.eye(6,dtype=torch.float64);cert=certificate(Q,K)
    assert cert['relative_anisotropic_residual']<1e-12
    scalar=((u@Q)*u).sum(-1);pred=cert['slope']*direct+cert['isotropic']*u.square().sum(-1)
    equality_error=float((scalar-pred).abs().max());assert equality_error<1e-10
    perturb=torch.randn(6,6,generator=g,dtype=torch.float64);perturb=(perturb+perturb.T)/2
    broken=certificate(Q+.1*perturb,K);assert broken['relative_anisotropic_residual']>.01
    degenerate=certificate(Q,torch.eye(6,dtype=torch.float64));assert not degenerate['identified']
    return dict(passed=True,gram_max_abs=gram_error,planted_equality_max_abs=equality_error,planted_certificate=cert,perturbed_certificate=broken,isotropic_reader_rejected=True,scope='Exact equality q=a*spread+b*norm²+beta requires zero matrix residual; finite-sample fit does not certify it. Spread is not entropy. No trained-model result yet.')
