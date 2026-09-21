"""Minimize the largest convex quadratic; retain solver and KKT diagnostics."""
import numpy as np
from scipy.optimize import minimize,nnls

def solve(G,b,c,initial=None):
 G=np.asarray(G,dtype=float);b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float)
 assert G.ndim==3 and b.shape==G.shape[:2] and c.shape==(len(G),)
 assert np.max(np.abs(G-G.swapaxes(-1,-2)))<1e-9
 assert np.linalg.eigvalsh(G).min()>-1e-9
 n=G.shape[1]
 def values(x):return np.einsum('i,kij,j->k',x,G,x)+2*b@x+c
 def gradients(x):return 2*np.einsum('kij,j->ki',G,x)+2*b
 x=np.zeros(n) if initial is None else np.asarray(initial,dtype=float)
 def constraints(z):return z[-1]-values(z[:-1])
 def derivative(z):return np.column_stack([-gradients(z[:-1]),np.ones(len(G))])
 result=minimize(lambda z:z[-1],np.r_[x,values(x).max()],jac=lambda z:np.r_[np.zeros(n),1.],constraints=[dict(type='ineq',fun=constraints,jac=derivative)],method='SLSQP',options=dict(ftol=1e-12,maxiter=1000))
 x=result.x[:-1];q=values(x);maximum=float(q.max());active=np.flatnonzero(maximum-q<1e-7*max(1,abs(maximum)))
 K=np.vstack([gradients(x)[active].T,np.ones(len(active))]);weights,residual=nnls(K,np.r_[np.zeros(n),1.],maxiter=10000)
 return dict(x=x.tolist(),maximum=maximum,epigraph=float(result.x[-1]),values=q.tolist(),success=bool(result.success),message=result.message,iterations=int(result.nit),primal_violation=float(max(0,maximum-result.x[-1])),active=active.tolist(),multipliers=weights.tolist(),kkt_residual=float(residual))
