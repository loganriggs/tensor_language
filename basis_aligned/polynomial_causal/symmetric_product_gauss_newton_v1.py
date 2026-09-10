"""Exact matrix-free joint Gauss-Newton for partially symmetric product sums.

Coordinates are unnormalized A[q,d], B[q,d], Z[o,q], in a whitened output
metric. The augmented residual includes sqrt(penalty)*each component tensor.
No target tensor or full Jacobian/Hessian is materialized. This is joint GN,
not an incorrect reduced Hessian obtained by detaching conditional writers.
"""
import torch
from joint_quadratic_fit_v1 import product_cross


def pack(parts):
    return torch.cat([p.reshape(-1) for p in parts])


def unpack(vector,reference):
    result=[];offset=0
    for p in reference:
        result.append(vector[offset:offset+p.numel()].reshape(p.shape));offset+=p.numel()
    if offset!=vector.numel():raise ValueError('Wrong joint vector size')
    return tuple(result)


def adjoint(parameters,terms,paired=False):
    """J(parameters)^T tensor(terms); paired isolates aligned components.

For paired mode target terms have k*q columns ordered in repeated q blocks.
"""
    a,b,z=parameters;c,d,y=terms
    gram=product_cross(a,b,c,d);output=z.T@y
    if paired:
        q=len(a)
        if len(c)%q:raise ValueError('Paired target terms must repeat q columns')
        mask=torch.eye(q,device=a.device,dtype=a.dtype).repeat(1,len(c)//q)
        gram=gram*mask;output=output*mask
    ga=.5*((output*(b@d.T))@c+(output*(b@c.T))@d)
    gb=.5*((output*(a@d.T))@c+(output*(a@c.T))@d)
    gz=y@gram.T
    return ga,gb,gz


def tangent(parameters,direction):
    a,b,z=parameters;da,db,dz=direction
    return torch.cat([a,a,da]),torch.cat([b,db,b]),torch.cat([dz,z,z],dim=1)


def normal_action(parameters,vector,penalty=0.):
    direction=unpack(vector,parameters);terms=tangent(parameters,direction)
    full=pack(adjoint(parameters,terms))
    if penalty:full=full+penalty*pack(adjoint(parameters,terms,paired=True))
    return full


def normal_diagonal(parameters,penalty=0.):
    a,b,z=parameters;zn=z.square().sum(0)[:,None]
    da=.5*zn*(b.square().sum(1,keepdim=True)+b.square())
    db=.5*zn*(a.square().sum(1,keepdim=True)+a.square())
    g=.5*(a.square().sum(1)*b.square().sum(1)+(a*b).sum(1).square())
    dz=g[None,:].expand_as(z)
    return (1+penalty)*pack((da,db,dz))


def loss_gradient(parameters,target,total,penalty=0.):
    a,b,z=parameters;l,r,d=target
    gram=product_cross(a,b,a,b);cross=product_cross(a,b,l,r)
    gramout=z.T@z;crossout=z.T@d
    error=(total+(gram*gramout).sum()-2*(cross*crossout).sum())/total
    energy=(gram.diag()*gramout.diag()).sum()/total
    g=pack(adjoint(parameters,parameters))-pack(adjoint(parameters,target))
    if penalty:g=g+penalty*pack(adjoint(parameters,parameters,paired=True))
    # Return half-loss gradient, matching J^T J conventions. Full loss is
    # reconstruction + penalty; its gradient is twice this vector.
    return error+penalty*energy,g/total,dict(reconstruction=error,component_energy=energy)
