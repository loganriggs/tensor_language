"""Output-sharing block forms over an existing shared quadratic dictionary.
Fit output subspace to calibration predictions in a fixed Euclidean output frame,
then diagonalize each root quadratic form exactly. Leaves remain unchanged.
This is a restricted graph candidate, not arbitrary DAG search or semantic recovery.
"""
import torch
from empirical_quartic_dictionary import features


def compile_root(U,V,C,output_metric,x,rank):
    m=U.shape[0];i,j=torch.triu_indices(m,m,device=U.device)
    folded=output_metric@C;prediction=features(x,U,V)@folded.T
    _,_,vh=torch.linalg.svd(prediction,full_matrices=False)
    if not 1<=rank<=min(prediction.shape):raise ValueError('invalid output rank')
    return compile_basis(U,V,C,output_metric,vh[:rank].T)


def compile_basis(U,V,C,output_metric,basis):
    m=U.shape[0];rank=basis.shape[1];i,j=torch.triu_indices(m,m,device=U.device)
    root=basis.T@(output_metric@C)
    forms=root.new_zeros(rank,m,m)
    forms[:,i,j]=root/torch.where(i==j,1.,2.)
    forms[:,j,i]=forms[:,i,j]
    eigen,vectors=torch.linalg.eigh(forms)
    writer=torch.linalg.solve(output_metric,basis)
    return dict(U=U,V=V,root_eigenvalues=eigen,root_eigenvectors=vectors,writer=writer)


def expand_root_writer(program):
    E=program['root_eigenvectors'];values=program['root_eigenvalues'];forms=(E*values[:,None,:])@E.transpose(-1,-2)
    i,j=torch.triu_indices(forms.shape[1],forms.shape[2],device=forms.device)
    coefficients=forms[:,i,j]*torch.where(i==j,1.,2.)
    return program['writer']@coefficients


def evaluate(program,x):
    U,V=program['U'],program['V'];m,k,_=U.shape
    q=((x@U.flatten(0,1).T)*(x@V.flatten(0,1).T)).reshape(len(x),m,k).sum(2)
    projected=torch.einsum('ni,gij->ngj',q,program['root_eigenvectors'])
    h=(projected.square()*program['root_eigenvalues']).sum(-1)
    return h@program['writer'].T


def price(program):
    m,k,d=program['U'].shape;r=len(program['root_eigenvalues']);v=program['writer'].shape[0]
    return dict(products=m*k+r*m,additions=2*m*k*(d-1)+m*(k-1)+r*m*(m-1)+r*(m-1)+v*(r-1),stored_coefficients=sum(t.numel() for t in program.values()),quadratic_features=m,output_shared_forms=r)


def controls():
    torch.manual_seed(5400);torch.set_default_dtype(torch.float64);rows=[]
    for family in ['independent','shared_input','shared_output','squares','cancellation']:
        U,V=torch.randn(3,2,4),torch.randn(3,2,4);C=torch.randn(3,6);rank=3
        if family=='shared_input':U[1]=U[0]
        if family=='shared_output':C=torch.randn(3,1)@torch.randn(1,6);rank=1
        if family=='squares':V=U.clone()
        if family=='cancellation':U[2]=U[0];V[2]=-V[0]
        metric=torch.randn(3,3)+3*torch.eye(3);x=torch.randn(71,4);fresh=torch.randn(503,4);program=compile_root(U,V,C,metric,x,rank);truth=features(fresh,U,V)@C.T;error=float(((evaluate(program,fresh)-truth)@metric.T).norm()/(truth@metric.T).norm());assert error<1e-12
        rows.append(dict(family=family,rank=rank,fresh_replay=error,price=price(program)))
    return rows
if __name__=='__main__':
    import json
    from pathlib import Path
    torch.set_num_threads(2);rows=controls();Path(__file__).with_name('SHARED_ROOT_BLOCK_CONTROLS_V1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
