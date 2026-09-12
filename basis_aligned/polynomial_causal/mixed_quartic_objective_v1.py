"""Variable projection on a fixed mixed-edge graph; exact target oracle supplied."""
import torch
from quartic_selected_edges_v1 import gram,target_cross
from quartic_manifold_cg_v1 import tangent

def make_objective(edges,oracle,target_energy=0.):
    def evaluate(b,n,divisor,gradient=False):
        if gradient:b=b.detach().requires_grad_();n=n.detach().requires_grad_()
        with torch.set_grad_enabled(gradient):
            k=gram(b,n,edges);c=target_cross(b,n,edges,oracle)
            with torch.no_grad():mix=torch.linalg.solve(k,c)
            loss=(target_energy+(mix*(k@mix)).sum()-2*(mix*c).sum())/divisor
            if gradient:
                gb,gn=tangent(b,n,*torch.autograd.grad(loss,(b,n)))
                return float(loss.detach()),mix.detach(),gb.detach(),gn.detach()
            return float(loss),mix
    return evaluate
