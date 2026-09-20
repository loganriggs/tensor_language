"""Exact Gaussian quadratic inner product in whitened coordinates, with nonzero mean."""
from implicit_quadratic import inner

def gaussian_inner(c,a,b,d,l,r,mean=None):
 base=inner(c,a,b,d,l,r)
 if mean is None:return base
 am,bm,lm,rm=a@mean,b@mean,l@mean,r@mean
 cf=c@(am*bm);cg=d@(lm*rm)
 tf=c@(a*b).sum(-1);tg=d@(l*r).sum(-1)
 lf=(c*am)@b+(c*bm)@a;lg=(d*lm)@r+(d*rm)@l
 return base+(lf*lg).sum()+(cf*tg+cg*tf+cf*cg).sum()
