"""Exact block least squares for symmetric quadratic CP coefficients, not unsymmetric CP ALS."""
import torch
from core import packed_quadratic,normrows

def sweep(C,A,B,target,chol):
 o,k=C.shape;d=A.shape[1];basis=torch.eye(k*d,dtype=A.dtype).reshape(k*d,k,d);rhs=(target@chol).flatten()
 phi=packed_quadratic(A,B);C=torch.linalg.lstsq((phi@chol).T,(target@chol).T,driver='gelsd').solution.T
 design=(torch.einsum('ok,pkm->pom',C,packed_quadratic(basis,B))@chol).reshape(k*d,-1).T
 A=torch.linalg.lstsq(design,rhs,driver='gelsd').solution.reshape(k,d);norm=A.norm(dim=1).clamp_min(1e-30);A=A/norm[:,None];C=C*norm
 design=(torch.einsum('ok,pkm->pom',C,packed_quadratic(A,basis))@chol).reshape(k*d,-1).T
 B=torch.linalg.lstsq(design,rhs,driver='gelsd').solution.reshape(k,d);norm=B.norm(dim=1).clamp_min(1e-30);B=B/norm[:,None];C=C*norm
 return C,A,B
