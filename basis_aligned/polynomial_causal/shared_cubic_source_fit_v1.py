"""Product-of-spheres adapter for the existing safeguarded manifold L-BFGS."""
import torch
from shared_cubic_source_projection_v1 import capture
from quartic_manifold_lbfgs_v1 import fit as manifold_fit
from quartic_manifold_cg_v1 import tangent

def fit(atoms,weights,divisor,max_steps=512,max_seconds=300,callback=None):
    rank,_,dimension=atoms.shape
    b=atoms.reshape(rank*3,dimension,1);b=b/b.norm(dim=1,keepdim=True)
    n=torch.ones(rank*3,1,device=b.device,dtype=b.dtype)
    def evaluate(b,n,divisor,gradient):
        if not gradient:
            try:
                with torch.no_grad():value=-capture(b[...,0].reshape(rank,3,dimension),*weights)/divisor
                return float(value),None
            except torch.linalg.LinAlgError:return float('inf'),None
        with torch.enable_grad():
            bb=b.detach().requires_grad_(True);value=-capture(bb[...,0].reshape(rank,3,dimension),*weights)/divisor
            gb=torch.autograd.grad(value,bb)[0]
        gb,gn=tangent(b,n,gb,torch.zeros_like(n))
        return float(value.detach()),None,gb,gn
    bb,_,_,history,reason=manifold_fit(b,n,evaluate,divisor,max_steps,max_seconds,callback)
    return bb[...,0].reshape(rank,3,dimension),history,reason
