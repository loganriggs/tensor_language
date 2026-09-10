"""Explicit scale-invariant component-energy penalty, not a silent ridge repair."""
import torch
from structured_quadratic_models_v1 import QuadraticObjective

class EnergyRegularizedObjective(QuadraticObjective):
    def __init__(self,*args,penalty,**kwargs):
        super().__init__(*args,**kwargs)
        if penalty<=0:raise ValueError('Positive explicitly registered penalty required')
        self.penalty=float(penalty)

    def terms(self,model):
        cross,gram=self.cross_gram(model)
        # This is the exact conditional optimum of residual error plus lambda
        # times sum_j ||U w_j||^2 ||feature_j||^2. The same output metric weights
        # both terms, so it cancels from the normal equation for the writers.
        regularized_gram=gram+self.penalty*torch.diag(gram.diag())
        with torch.no_grad():w=torch.linalg.solve(regularized_gram,cross.T).T
        output_gram=w.T@self.metric@w
        residual=(self.total+(output_gram*gram).sum()-2*((self.metric@cross)*w).sum())/self.total
        component_energy=(output_gram.diag()*gram.diag()).sum()/self.total
        return residual+self.penalty*component_energy,residual,component_energy,w,gram,regularized_gram

    def loss(self,model):
        loss,_,_,w,gram,_=self.terms(model)
        return loss,w,gram

    def diagnostics(self,model):
        model.zero_grad(set_to_none=True)
        loss,residual,energy,w,gram,regularized=self.terms(model);loss.backward()
        captured=max(1-float(residual.detach()),1e-12)
        stationarity=max(float(p.grad.norm()*p.detach().norm().clamp_min(1))/captured for p in model.parameters())
        joint_energy=((w.T@self.metric@w)*gram).sum()/self.total
        return dict(squared_relative_error=float(residual.detach()),optimization_loss=float(loss.detach()),
            component_energy_over_native_total=float(energy.detach()),penalty=self.penalty,
            cancellation_ratio=float((energy/joint_energy).detach()),captured_energy_fraction=captured,
            relative_stationarity=stationarity,gradient_max_abs=max(float(p.grad.abs().max()) for p in model.parameters()),
            gram_condition=float(torch.linalg.cond(gram.detach())),
            regularized_gram_condition=float(torch.linalg.cond(regularized.detach()))),w.detach()
