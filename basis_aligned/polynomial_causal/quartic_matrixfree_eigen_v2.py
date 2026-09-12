"""Symmetric packed coordinates and bounded real-symmetric ARPACK adapter.

Off-diagonal coordinates carry sqrt(2), preserving the Frobenius metric.
Largest-magnitude Ritz pairs are initialization proposals, not quadratic atoms.
"""
import math
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator,eigsh,ArpackNoConvergence

class SymmetricCoordinates:
    def __init__(self,dimension,device='cpu'):
        self.dimension=dimension;self.device=device
        self.indices=torch.triu_indices(dimension,dimension,device=device)
        self.scale=torch.full((self.indices.shape[1],),math.sqrt(2.),dtype=torch.float64,device=device)
        self.scale[self.indices[0]==self.indices[1]]=1.
        self.size=len(self.scale)
    def pack(self,matrix):
        return matrix[self.indices[0],self.indices[1]]*self.scale
    def unpack(self,vector):
        matrix=torch.zeros(self.dimension,self.dimension,dtype=torch.float64,device=self.device)
        values=vector/self.scale
        matrix[self.indices[0],self.indices[1]]=values
        matrix[self.indices[1],self.indices[0]]=values
        return matrix

class ActionBudgetExceeded(RuntimeError):pass

@torch.no_grad()
def eigenmatrices(action,dimension,device='cpu',k=2,seed=91951,tol=1e-6,
                  maxiter=50,ncv=12,max_actions=200):
    coordinates=SymmetricCoordinates(dimension,device)
    assert 0<k<coordinates.size and max_actions>0
    calls=0
    def matvec(vector):
        nonlocal calls
        if calls>=max_actions:raise ActionBudgetExceeded('operator action budget reached')
        calls+=1
        q=coordinates.unpack(torch.as_tensor(vector,dtype=torch.float64,device=device))
        y=action(q);y=(y+y.T)/2
        return coordinates.pack(y).cpu().numpy()
    op=LinearOperator((coordinates.size,coordinates.size),matvec=matvec,dtype=np.float64)
    v0=np.random.default_rng(seed).standard_normal(coordinates.size)
    status='converged'
    try:
        values,vectors=eigsh(op,k=k,which='LM',v0=v0,ncv=min(ncv,coordinates.size),maxiter=maxiter,tol=tol)
    except ArpackNoConvergence as error:
        status='partial';values,vectors=error.eigenvalues,error.eigenvectors
    except ActionBudgetExceeded:
        status='action_limit';values=np.empty(0);vectors=np.empty((coordinates.size,0))
    matrices=[];residuals=[]
    for value,vector in zip(values,vectors.T):
        q=coordinates.unpack(torch.as_tensor(vector,dtype=torch.float64,device=device))
        y=action(q)
        residuals.append(float((y-float(value)*q).norm()/max(float(y.norm()),abs(float(value)),1e-30)))
        matrices.append(q)
    return values,matrices,dict(status=status,operator_actions=calls,
        verification_actions=len(matrices),relative_eigen_residuals=residuals,
        packed_dimension=coordinates.size,requested_eigenpairs=k,
        lanczos_vectors=min(ncv,coordinates.size),seed=seed)
