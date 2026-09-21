"""Exact numerical common-product representation for regular symmetric pairs.

A simple real/complex eigenspace construction, with fail-closed replay and
conditioning checks. Defective pencils are unsupported. This is not a general
canonical-form algorithm or an approximate joint diagonalizer.
"""
import torch

def compile_pair(A,B,tol=1e-10):
    assert A.ndim==2 and A.shape==B.shape and A.shape[0]==A.shape[1]
    d=len(A);assert A.dtype==torch.float64 and A.device.type=='cpu'
    # A generic fixed invertible pencil base also handles singular A or B.
    C=A+B;D=A-.37*B
    condition=float(torch.linalg.cond(C))
    if not condition<1e10:raise ValueError('Ill-conditioned/singular pencil base')
    pencil=torch.linalg.solve(C,D)
    values,V=torch.linalg.eig(pencil)
    eigcond=float(torch.linalg.cond(V))
    if not eigcond<1e9:raise ValueError('Defective or ill-conditioned eigenbasis')
    used=set();columns=[];blocks=[];i=0
    for j in range(d):
        if j in used:continue
        value=values[j];threshold=1e-7*(1+float(value.abs()))
        if abs(float(value.imag))<=threshold:
            group=[k for k in range(d) if k not in used and abs(float(values[k].imag))<=threshold and float((values[k].real-value.real).abs())<=threshold]
            # Within a repeated real eigenvalue, diagonalize its symmetric
            # pencil-base form. Euclidean QR removes arbitrary eigensolver bases.
            # A repeated real eigenvalue can be returned as tiny complex pairs.
            # Taking real parts can duplicate columns: recover the real nullspace.
            center=values[group].real.mean()
            if len(group)==1:
                E=V[:,group].real
                E=E/E.norm(dim=0,keepdim=True)
            else:
                _,singular,right=torch.linalg.svd(pencil-center*torch.eye(d,dtype=A.dtype))
                E=right[-len(group):].T
            if float((pencil@E-center*E).norm()/pencil.norm())>1e-7:
                raise ValueError('Inaccurate real eigenspace')
            _,rotation=torch.linalg.eigh(E.T@C@E);E=E@rotation
            for k in range(len(group)):columns.append(E[:,k]);blocks.append([i]);i+=1
            used.update(group)
        else:
            if value.imag<0:continue
            partner=int((values-value.conj()).abs().argmin());assert partner!=j
            columns.extend([V[:,j].real,V[:,j].imag]);blocks.append([i,i+1]);i+=2;used.update([j,partner])
    if i!=d:raise ValueError('Incomplete block basis')
    X=torch.stack(columns,1);cond=float(torch.linalg.cond(X))
    if not cond<1e9:raise ValueError('Ill-conditioned real block basis')
    transformed=[X.T@Q@X for Q in [A,B]];left=[];right=[];types=[];weights=[]
    for block in blocks:
        i=block[0]
        if len(block)==1:
            left.append(i);right.append(i);types.append(0);weights.append(torch.stack([Q[i,i] for Q in transformed]))
        else:
            j=block[1]
            # Re/Im eigenvectors give traceless symmetric2x2 forms:
            # alpha*(u^2-v^2) + beta*u*v. Factor u^2-v^2 explicitly.
            left.extend([i,i]);right.extend([j,j]);types.extend([1,2]);weights.extend([torch.stack([.5*(Q[i,i]-Q[j,j]) for Q in transformed]),torch.stack([2*Q[i,j] for Q in transformed])])
    coefficients=torch.stack(weights);indices=torch.tensor([left,right,types],dtype=torch.int64);inverse=torch.linalg.inv(X);reconstructed=[]
    for output in range(2):
        G=torch.zeros_like(A)
        for pos,(i,j,kind) in enumerate(zip(left,right,types)):
            w=coefficients[pos,output]
            if kind==0:G[i,i]+=w
            elif kind==1:G[i,i]+=w;G[j,j]-=w
            else:G[i,j]+=w/2;G[j,i]+=w/2
        reconstructed.append(inverse.T@G@inverse)
    errors=[float((q-r).norm()/q.norm()) for q,r in zip([A,B],reconstructed)]
    if max(errors)>tol:raise ValueError(f'Pair reconstruction failed: {errors}')
    return dict(input_transform=inverse.T,product_indices=indices,product_weights=coefficients,blocks=blocks,diagnostics=dict(pencil_condition=condition,eigenbasis_condition=eigcond,real_basis_condition=cond,matrix_replay=errors))

def products(t,indices):
    i,j,kind=indices.long();u=t[...,i];v=t[...,j]
    left=torch.where(kind==1,u+v,u);right=torch.where(kind==1,u-v,v)
    return left*right
