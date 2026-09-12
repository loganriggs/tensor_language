"""Extract the shared parent, two child readers, and private query writers."""
import torch
from shared_cubic_source_projection_v1 import atom_gram

def source_ports(source,atoms):
    assert torch.equal(atoms[-2,:2],atoms[-1,:2])
    parent=(source@atoms[-2,0])*(source@atoms[-2,1])
    children=source@atoms[-2:,2].T
    return parent,children

def private_writers(query,atoms,factors):
    a,b,c=factors
    qa=torch.einsum('nd,hrtd->nhrt',query,a);qb=torch.einsum('nd,hrtd->nhrt',query,b)
    cross=torch.einsum('nhrt,hrto->nhro',qa*qb,c);shape=cross.shape
    rhs=cross.permute(2,0,1,3).flatten(1)
    beta=torch.linalg.solve(atom_gram(atoms),rhs).reshape(shape[2],shape[0],shape[1],shape[3]).permute(1,2,0,3)
    return beta[:,:,-2:]
