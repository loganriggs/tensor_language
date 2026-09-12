"""Execute selected source features with exact private query writers."""
import torch
from shared_cubic_source_projection_v1 import atom_gram

def execute(query,source,components,mix,factors):
    a,b,c=factors
    qa=torch.einsum('nd,hctd->nhct',query,a);qb=torch.einsum('nd,hctd->nhct',query,b)
    cross=torch.einsum('nhct,hcto->nhco',qa*qb,c)
    cross=torch.einsum('rc,nhco->nhro',mix,cross)
    gram=mix@atom_gram(components)@mix.T
    shape=cross.shape;rhs=cross.permute(2,0,1,3).flatten(1)
    beta=torch.linalg.solve(gram,rhs).reshape(shape[2],shape[0],shape[1],shape[3]).permute(1,2,0,3)
    phi=torch.einsum('nd,cid->nci',source,components).prod(-1)@mix.T
    return phi[:,None,:,None]*beta
