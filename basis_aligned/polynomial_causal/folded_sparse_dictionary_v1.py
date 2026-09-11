"""Exact full folded loss through shared sparse input features.

Reuse the existing chunked CP factor gradient. Native and candidate writers
must already include the unembedding metric. Fixed support values and shared
dictionary directions can vary jointly. No reader proxy, text or L1 penalty.
"""
import torch
from chunked_bilinear_coefficient_v1 import value_gradient


def decode(raw_basis,indices,values,row_scale):
    """Values are normalized-reader coordinates; return physical readers."""
    norms=raw_basis.norm(dim=1,keepdim=True)
    if not bool((norms>1e-12).all()):raise ArithmeticError('Vanishing dictionary parameter row')
    basis=raw_basis/norms
    rows=torch.arange(len(indices),device=indices.device)[:,None].expand_as(indices)
    code=torch.sparse_coo_tensor(torch.stack((rows.flatten(),indices.flatten())),
        (values*row_scale[:,None]).flatten(),(len(indices),len(basis))).coalesce()
    readers=torch.sparse.mm(code,basis)
    return readers,basis,code,norms


@torch.no_grad()
def loss_gradient(native,writer,raw_basis,indices,values,row_scale,total,chunk=256):
    readers,basis,code,norms=decode(raw_basis,indices,values,row_scale)
    a,b=readers.chunk(2)
    loss,(ga,gb,_),details=value_gradient(*native,a,b,writer,total,penalty=0.,chunk=chunk)
    reader_gradient=torch.cat((ga,gb))
    # This temporary is rows x dictionary, not rows x support x input dimension.
    all_codes=(reader_gradient*row_scale[:,None])@basis.T
    value_grad=all_codes.gather(1,indices)
    basis_grad=torch.sparse.mm(code.transpose(0,1),reader_gradient)
    tangent=basis_grad-(basis_grad*basis).sum(1,keepdim=True)*basis
    raw_grad=tangent/norms
    denominator=loss.abs().clamp_min(1e-12)
    details.update(dictionary_relative_stationarity=float(tangent.norm()*len(basis)**.5/denominator),
        codes_relative_stationarity=float(value_grad.norm()*values.norm()/denominator),
        raw_basis_norm_min=float(norms.min()),raw_basis_norm_max=float(norms.max()))
    return loss,(raw_grad,value_grad),details
