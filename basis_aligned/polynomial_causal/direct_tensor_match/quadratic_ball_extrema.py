"""Dense secular trust-region solver with PSD, complementarity and dual-gap checks.
Minimize x.Q.x + l.x + b over ||x|| <= radius. Float64 numerical certificate,
not interval arithmetic. Handles interior and hard-case boundary solutions.
"""
import torch

def minimum(Q,l,b,radius):
 Q=(Q+Q.T)/2;e,V=torch.linalg.eigh(Q);g=(V.T@l)/2;scale=max(1.,float(e.abs().max()),float(g.norm()/radius));tol=1e-12*scale;bound=max(0.,-float(e[0]));den=e+bound;null=den.abs()<=tol
 hard=bool(g[null].norm()<=tol*max(radius,1.))
 y0=torch.zeros_like(g);y0[~null]=-g[~null]/den[~null]
 if hard and float(y0.norm())<=radius:
  lam=bound;y=y0.clone();case='interior' if lam==0 else 'hard'
  if lam>0:
   k=int(null.nonzero()[0]);y[k]=(max(0.,radius**2-float(y.square().sum())))**.5
 else:
  lo=bound;hi=bound+scale+float(g.norm()/radius)
  def norm_at(v):return float((g/(e+v)).norm())
  while norm_at(hi)>radius:hi=bound+2*(hi-bound)
  for _ in range(100):
   mid=(lo+hi)/2
   if mid==lo or mid==hi:break
   if norm_at(mid)>radius:lo=mid
   else:hi=mid
  lam=hi;y=-g/(e+lam);case='secular'
 x=V@y;value=x@Q@x+l@x+b;shift=e+lam;safe=shift.abs()>tol
 dual=float(b-(g[safe].square()/shift[safe]).sum()-lam*radius**2)
 residual=float((Q@x+l/2+lam*x).norm())/(scale*max(radius,1.));gap=abs(float(value)-dual)/max(abs(float(value)),abs(dual),1.)
 certificate=dict(stationarity=residual,relative_dual_gap=gap,shift_min_eigenvalue=float(shift.min()),radius_excess=max(0.,float(x.norm())-radius),complementarity=abs(lam*(float(x.square().sum())-radius**2))/(scale*max(radius**2,1.)))
 assert residual<1e-8 and gap<1e-8 and certificate['radius_excess']<1e-7*max(radius,1.) and certificate['complementarity']<1e-8,certificate
 return dict(value=float(value),x=x,multiplier=lam,case=case,certificate=certificate)

def extrema(Q,l,b,radius):
 lo=minimum(Q,l,b,radius);negative=minimum(-Q,-l,-b,radius);hi=dict(negative,value=-negative['value'])
 return dict(minimum=lo,maximum=hi,worst_absolute=max(abs(lo['value']),abs(hi['value'])))

def controls():
 dtype=torch.float64;cases=[
 ('positive_interior',torch.diag(torch.tensor([1.,2.],dtype=dtype)),torch.tensor([-1.,0.],dtype=dtype),0.,2.,-.25),
 ('hard_negative',torch.diag(torch.tensor([-2.,1.],dtype=dtype)),torch.zeros(2,dtype=dtype),0.,3.,-18.),
 ('linear',torch.zeros(2,2,dtype=dtype),torch.tensor([3.,4.],dtype=dtype),2.,2.,-8.),
 ('singular_interior',torch.diag(torch.tensor([0.,1.],dtype=dtype)),torch.tensor([0.,-1.],dtype=dtype),0.,2.,-.25),
 ('indefinite_easy',torch.diag(torch.tensor([-2.,1.],dtype=dtype)),torch.tensor([1.,0.],dtype=dtype),0.,2.,-10.),
 ('hard_nonzero_linear',torch.diag(torch.tensor([-2.,1.],dtype=dtype)),torch.tensor([0.,-2.],dtype=dtype),0.,2.,-8.-1/3)]
 result=[]
 for name,Q,l,b,r,value in cases:
  row=minimum(Q,l,b,r);assert abs(row['value']-value)<1e-10,(name,row,value)
  gen=torch.Generator().manual_seed(919);U=torch.linalg.qr(torch.randn(2,2,generator=gen,dtype=dtype))[0];rotated=minimum(U@Q@U.T,U@l,b,r);assert abs(rotated['value']-value)<1e-10
  result.append(dict(name=name,value=row['value'],case=row['case'],certificate=row['certificate']))
 return result
