"""Unit-atom oblique reader maps and exact encoding on selected supports."""
import torch
from tyler_reader_shape_v1 import reader_maps


def canonical_maps(rotation,shape):
    maps=reader_maps(rotation,shape)
    norms=maps['synthesis'].norm(dim=0)
    return dict(dictionary=maps['synthesis']/norms,
                weight_coder=maps['weight_coder']*norms[:,None],
                input_features=maps['input_features']/norms[:,None])


@torch.no_grad()
def encode(dictionary,weight_coder,readers,k,chunk=64):
    """Readers are Nxd rows. Supports use full coordinates; values use LS."""
    coordinates=readers@weight_coder.T
    indices=coordinates.abs().topk(k,dim=1).indices
    truncated=coordinates.gather(1,indices)
    values=[];residuals=[];before=0.;after=0.
    for start in range(0,len(readers),chunk):
        sl=slice(start,start+chunk)
        atoms=dictionary.T[indices[sl]]
        gram=atoms@atoms.transpose(-1,-2)
        rhs=(atoms@readers[sl,:,None])
        chol=torch.linalg.cholesky((gram+gram.transpose(-1,-2))/2)
        fitted=torch.cholesky_solve(rhs,chol).squeeze(-1)
        errors=(gram@fitted[:,:,None]-rhs)
        residuals.append(float(errors.norm()/rhs.norm().clamp_min(1e-30)))
        values.append(fitted)
        old=torch.einsum('nk,nkd->nd',truncated[sl],atoms)
        new=torch.einsum('nk,nkd->nd',fitted,atoms)
        before+=float((old-readers[sl]).square().sum())
        after+=float((new-readers[sl]).square().sum())
    return indices,torch.cat(values),dict(normal_equation_relative_residual=max(residuals),
        truncation_squared_error=before,refit_squared_error=after,
        aggregate_relative_gain=(before-after)/max(before,1e-30),
        scope='Exact values for fixed coordinate-threshold supports; support selection is not globally optimal.')
