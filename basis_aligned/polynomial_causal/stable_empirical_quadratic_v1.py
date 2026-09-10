"""QR variable projection, whitened output residual, envelope first gradient.

No differentiation through QR. This is not a reduced-Hessian implementation.
All feature columns retained; no hidden ridge or truncated singular spectrum.
"""
import torch

def qr_writers(features,targets):
    """Solve min_B ||features B-targets||_F for full column rank features."""
    q,r=torch.linalg.qr(features,mode='reduced')
    return torch.linalg.solve_triangular(r,q.T@targets,upper=True),r

class StableEmpiricalObjective:
    def __init__(self,metric,x,y):
        self.x=x;self.y=y;self.metric=metric
        self.output_root=torch.linalg.cholesky(metric)
        self.target=y@self.output_root
        self.total=self.target.square().sum()

    def features(self,model):
        a,b,c=model.components()
        return ((self.x@a.T)*(self.x@b.T))@c

    def loss(self,model):
        f=self.features(model)
        with torch.no_grad():
            beta,r=qr_writers(f.detach(),self.target)
            writer=torch.linalg.solve_triangular(self.output_root.T,beta.T,upper=True)
        residual=f@beta-self.target
        loss=residual.square().sum()/self.total
        return loss,writer,r.T@r

    def diagnostics(self,model):
        model.zero_grad(set_to_none=True);loss,w,g=self.loss(model);loss.backward()
        captured=max(1-float(loss.detach()),1e-12)
        station=max(float(p.grad.norm()*p.detach().norm().clamp_min(1))/captured for p in model.parameters())
        return dict(squared_relative_error=float(loss.detach()),optimization_loss=float(loss.detach()),captured_energy_fraction=captured,relative_stationarity=station,gradient_max_abs=max(float(p.grad.abs().max()) for p in model.parameters()),gram_condition=float(torch.linalg.cond(g.detach()))),w.detach()
