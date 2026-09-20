"""Gaussian conditional-pair diagnostic for a fixed dictionary of linear readers."""
import torch

def reader_frame(readers, transform):
    # x = mu + z @ transform.T; readers @ x have row space readers @ transform.
    _, values, vh = torch.linalg.svd(readers @ transform, full_matrices=False)
    rank = int((values > values[0] * 1e-10).sum())
    return vh[:rank].T, values

def paired_inputs(mu, transform, frame, count, generator):
    z = torch.randn(count, len(mu), generator=generator, device=mu.device, dtype=mu.dtype)
    a = torch.randn(z.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    b = torch.randn(z.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    shared = (z @ frame) @ frame.T
    a = shared + a - (a @ frame) @ frame.T
    b = shared + b - (b @ frame) @ frame.T
    return mu + a @ transform.T, mu + b @ transform.T

def toy_oracle():
    import numpy as np
    nodes, weights = np.polynomial.hermite.hermgauss(4)
    nodes=torch.tensor(nodes*2**.5);weights=torch.tensor(weights/np.pi**.5)
    grid=torch.cartesian_prod(*([torch.arange(4)]*5));z=nodes[grid];w=weights[grid].prod(1)
    # y=z0²+z1²+z0*z2, reader=z0. Conditional mean=z0²+1.
    f=z[:,0]**2+z[:,1]**2+z[:,0]*z[:,2]
    g=z[:,0]**2+z[:,3]**2+z[:,0]*z[:,4]
    irreducible=float((w*(f-g)**2).sum()/2)
    optimum=float((w*(f-z[:,0]**2-1)**2).sum())
    energy=float((w*f**2).sum())
    assert max(abs(irreducible-3),abs(optimum-3),abs(energy-9))<1e-12
    return dict(paired_difference_half=irreducible,conditional_mean_mse=optimum,target_energy=energy,relative_error_floor=(irreducible/energy)**.5)

if __name__=='__main__':
    import json
    from pathlib import Path
    r=toy_oracle();Path(__file__).with_name('CONDITIONAL_READER_ORACLE_V1.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
