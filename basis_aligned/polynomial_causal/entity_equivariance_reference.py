"""Paired KL lower bound for exactly entity-equivariant distribution predictors."""
import math
import torch


def information_radius(old_logits,aligned_new_logits):
    p=old_logits.log_softmax(-1);q=aligned_new_logits.log_softmax(-1)
    mixture=torch.logaddexp(p,q)-math.log(2)
    radius=.5*(p.exp()*(p-mixture)).sum(-1)+.5*(q.exp()*(q-mixture)).sum(-1)
    return radius,mixture


def controls():
    p=torch.tensor([[.7,.2,.1]],dtype=torch.float64).log();q=torch.tensor([[.2,.3,.5]],dtype=torch.float64).log()
    r=torch.tensor([[.1,.4,.5]],dtype=torch.float64).log();js,m=information_radius(p,q)
    kl=lambda a,b:(a.exp()*(a-b)).sum(-1)
    checks={'radius_identity':float((.5*kl(p,r)+.5*kl(q,r)-js-kl(m,r)).abs().max())<1e-12,
            'mixture_attains_bound':float((.5*kl(p,m)+.5*kl(q,m)-js).abs().max())<1e-12,
            'identical_zero':float(information_radius(p,p)[0].abs().max())<1e-12}
    permutation=torch.tensor([1,2,0]);inverse=permutation.argsort();renamed=p[:,inverse]
    checks['forward_vocabulary_alignment']=float(information_radius(p,renamed[:,permutation])[0].abs().max())<1e-12
    checks['wrong_inverse_alignment_live']=float(information_radius(p,renamed[:,inverse])[0].max())>.01
    peak=torch.tensor([[50.,-50.]],dtype=torch.float64)
    checks['disjoint_limit_log2']=abs(float(information_radius(peak,-peak)[0])-math.log(2))<1e-12
    return {'passed':all(checks.values()),'checks':checks}
