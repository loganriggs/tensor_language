"""Simple-pencil necessary test for simultaneous congruence blocks.
Not a general arithmetic-circuit impossibility test. CPU, double precision.
"""
import numpy as np
import torch
from scipy.sparse.csgraph import connected_components

def screen(Q, a, b, threshold=1e-8):
    A=torch.einsum('o,oij->ij',a,Q)
    B=torch.einsum('o,oij->ij',b,Q)
    cond=float(torch.linalg.cond(A))
    M=torch.linalg.solve(A,B)
    values,V=torch.linalg.eig(M)
    scale=max(float(values.abs().max()),1e-30)
    distances=(values[:,None]-values[None,:]).abs()
    distances.fill_diagonal_(float('inf'))
    gap=float(distances.min())/scale
    eigres=float((M.to(V.dtype)@V-V*values).norm()/M.norm())
    vcond=float(torch.linalg.cond(V))
    solve_res=float((A@M-B).norm()/B.norm())
    base=dict(anchor_condition=cond,eigenbasis_condition=vcond,relative_minimum_eigengap=gap,eigen_residual=eigres,solve_residual=solve_res,complex_eigenvalues=int((values.imag.abs()>1e-8*scale).sum()))
    if gap<=1e-10 or cond>=1e8 or vcond>=1e8 or max(eigres,solve_res)>=1e-10:
        return dict(**base,instrument='INCONCLUSIVE',reason='simple, adequately conditioned pencil not established')
    maps=torch.linalg.solve(A[None],Q)
    transformed=torch.linalg.solve(V[None],maps.to(V.dtype)@V)
    normalized=transformed/transformed.norm(dim=(-1,-2))[:,None,None]
    weights=normalized.abs().amax(0).numpy()
    weights=np.maximum(weights,weights.T);np.fill_diagonal(weights,0)
    components=[]
    for tol in [1e-10,threshold,1e-6]:
        count,labels=connected_components(weights>tol,directed=False)
        components.append(dict(threshold=tol,count=int(count),sizes=sorted(np.bincount(labels).tolist())))
    # Maximum spanning tree bottleneck: connectivity persists for cutoffs below this.
    n=len(weights);used=np.zeros(n,bool);used[0]=True;best=weights[0].copy();bottleneck=float('inf')
    for _ in range(n-1):
        best[used]=-1;j=int(best.argmax());bottleneck=min(bottleneck,float(best[j]));used[j]=True;best=np.maximum(best,weights[j])
    count,labels=connected_components(weights>threshold,directed=False)
    mask=torch.from_numpy(labels[:,None]!=labels[None,:])
    off=float(transformed[:,mask].norm()/transformed.norm())
    return dict(**base,instrument='PASS',connectivity=components,primary_components=int(count),primary_sizes=sorted(np.bincount(labels).tolist()),connectivity_bottleneck=bottleneck,off_block_transformed_operator_error=off,scalar_center_supported=bool(count==1),scope='Numerical necessary-condition screen. Connected simple-pencil commutation graph forces scalar center in exact arithmetic. Disconnected complex blocks are not automatically real congruence blocks; no real compiler is claimed.')
