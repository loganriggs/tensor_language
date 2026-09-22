"""Exact weight-derived compression of a fixed CP parent with movable factors."""
import torch
from quartic_cp import cp_gram
from mixed_gaussian_cp import gram_dynamic
from quartic_cp_profile import profile


def prepare(factors,coefficients,S,mu,lam):
    fw=[f@S for f in factors];bias=[f@mu for f in factors]
    with torch.no_grad():
        G=(gram_dynamic(fw,bias,fw,bias)+lam*cp_gram(factors,factors))/(1+lam)
        energy=(coefficients*(coefficients@G)).sum()
    return dict(factors=factors,coefficients=coefficients,transformed=fw,bias=bias,S=S,mu=mu,lam=lam,energy=energy)


def objective(parent,factors,ridge=1e-10):
    S,mu,lam=parent['S'],parent['mu'],parent['lam'];fw=[f@S for f in factors];bias=[f@mu for f in factors]
    G=(gram_dynamic(fw,bias,fw,bias)+lam*cp_gram(factors,factors))/(1+lam)
    cross=(gram_dynamic(parent['transformed'],parent['bias'],fw,bias)+lam*cp_gram(parent['factors'],factors))/(1+lam)
    X=parent['coefficients']@cross
    scale=G.diag().clamp_min(1e-30).sqrt();loss,unitC=profile(G/(scale[:,None]*scale[None,:]),X/scale,ridge=ridge)
    # Match the pruning objective's normalized-feature ridge, including its
    # dependence on the factor directions. This prevents a baseline metric change.
    C=unitC/scale
    return loss,C,dict(energy=parent['energy'],ridge_penalty=ridge*unitC.square().sum())
