"""Function tangent remaining after optimal infinitesimal writer compensation.

This quotients physical first-order function changes by output reweighting.
It is not the full variable-projection residual Jacobian/Hessian at nonzero
target residual and does not turn every null tangent into a finite symmetry.
"""
import torch
from conditional_writer_spectral_v1 import solve
from structured_branch_amplitudes_v1 import inner


@torch.no_grad()
def measure(a,b,w,da,db):
    tangent=(torch.cat((da,a)),torch.cat((b,db)),torch.cat((w,w),1))
    projection,_,_,solver=solve(tangent[2],tangent[0],tangent[1],a,b)
    projected=(a,b,projection)
    raw=float(inner(tangent,tangent));represented=float(inner(projected,projected))
    cross=float(inner(tangent,projected))
    remaining=raw+represented-2*cross
    return -projection,dict(raw_tangent_energy=raw,projected_energy=represented,
        remaining_energy=remaining,remaining_fraction=remaining/max(raw,1e-30),
        projection_energy_identity_error=abs(cross-represented)/max(raw,1e-30),writer_solve=solver)
