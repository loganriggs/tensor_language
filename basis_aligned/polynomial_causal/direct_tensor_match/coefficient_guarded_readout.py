"""Globally solve a fixed-feature quadratic fit under a coefficient-fit budget."""
import torch


class GuardedReadout:
    def __init__(self, coefficient_gram, coefficient_cross, target_gram, target_cross):
        # Grams include any explicitly chosen ridge. No hidden regularization.
        self.G0, self.X0 = coefficient_gram, coefficient_cross
        self.G, self.X = target_gram, target_cross
        self.L = torch.linalg.cholesky(self.G0)
        self.C0 = torch.cholesky_solve(self.X0.T, self.L).T
        temp = torch.linalg.solve_triangular(self.L, self.G, upper=False)
        H = torch.linalg.solve_triangular(self.L, temp.T, upper=False).T
        H = .5*(H+H.T)
        self.eigenvalues, self.E = torch.linalg.eigh(H)
        if not bool((self.eigenvalues > 0).all()):
            raise ValueError('Target Gram must be positive definite in coefficient coordinates')
        residual = self.X - self.C0 @ self.G
        self.B = torch.linalg.solve_triangular(self.L, residual.T, upper=False).T @ self.E
        self.capture = float((self.C0*self.X0).sum())

    def solve(self, budget):
        if budget < 0:
            raise ValueError('Budget must be nonnegative')
        if budget == 0:
            return self.C0.clone(), dict(multiplier=None, displacement=0., endpoint='coefficient')
        def displacement(lam):
            return float((self.B/(self.eigenvalues+lam)).square().sum())
        if displacement(0.) <= budget:
            lam = 0.
        else:
            low = 0.
            high = max(1.,float(self.B.norm())/budget**.5)
            for _ in range(90):
                middle = (low+high)/2
                if displacement(middle)>budget:low=middle
                else:high=middle
            lam=high
        Y=(self.B/(self.eigenvalues+lam))@self.E.T
        C=self.C0+torch.linalg.solve_triangular(self.L.T,Y.T,upper=True).T
        delta=C-self.C0
        actual=float((delta*(delta@self.G0)).sum())
        residual=C@self.G-self.X+lam*(C@self.G0-self.X0)
        denom=self.X.norm()+self.G.norm()*C.norm()+lam*(self.X0.norm()+self.G0.norm()*C.norm())
        return C,dict(multiplier=lam,displacement=actual,endpoint='target' if lam==0 else 'constrained',stationarity=float(residual.norm()/denom.clamp_min(1e-30)))
