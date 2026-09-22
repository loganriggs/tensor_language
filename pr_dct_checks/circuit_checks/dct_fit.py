"""Port of AJ's fitting loop onto the span interface.

Reproduces, line for line in substance, `dct.AsymmetricQuadraticDCT.fit` and
`sparse-asymmetric/pr_penalty_experiment.ParticipationRegularizedDCT.fit` from
ajskateboarder/redesigned-octo-couscous (reviewed in full before porting):

  * L, R: [d_source, F] unit columns; U: [d_target, F] unit columns; random normal init under torch.manual_seed(seed).
  * every iteration: L, R <- QR(L), QR(R); then for every context, per factor, the gradient of
        0.5 * (u . H[l, r])^2  -  w * S * PR(c[l, r]) / D          (w = 0 for the plain DCT)
    w.r.t. (u, l, r), where H is the forward-over-forward mixed Hessian of the target-position-averaged output and c the
    same mixed Hessian of the concatenated intermediate MLP hidden units (AJ's range: blocks source+1 .. target-1);
  * the averaged gradients replace the factors: U <- normalize(beta * gU + (1 - beta) * U), likewise L, R (beta = 1 default);
  * penalty scale S = (baseline fit's mean over contexts of the summed factor energy) / F, computed by the caller.

Differences from AJ's code: the context state is closed over explicitly (no nearest-neighbour lookup), and the forward pass is the
handoff adapter's, which check_adapter_parity.py shows to be exact against AJ's TensorMiddleSpan on a random model.
"""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch.func import grad, vmap

from .core import mixed_hessian, participation_ratio, concat_hidden


class SpanDCT:
    def __init__(self, num_factors, penalty_weight=0.0, penalty_scale=1.0):
        self.num_factors, self.penalty_weight, self.penalty_scale = num_factors, penalty_weight, penalty_scale

    def fit(self, make_span: Callable, contexts: Sequence, d_source: int, d_target: int, pr_layers: Sequence[int],
            max_iters=10, factor_batch=4, beta=1.0, seed=0, device="cuda", feature_dim=None):
        torch.manual_seed(seed)
        self.L = F.normalize(torch.randn(d_source, self.num_factors, device=device), dim=0)
        self.R = F.normalize(torch.randn(d_source, self.num_factors, device=device), dim=0)
        self.U = F.normalize(torch.randn(d_target, self.num_factors, device=device), dim=0)
        penalised = self.penalty_weight != 0
        if penalised and feature_dim is None:
            _, hidden = make_span(contexts[0])(torch.zeros(d_source, device=device), None)
            feature_dim = int(sum(hidden[k].numel() for k in pr_layers))
        self.feature_dim = feature_dim
        self.objective_values, self.score_energy_values, self.normalized_pr_values = [], [], []

        for _ in range(max_iters):
            with torch.no_grad():
                self.L, _ = torch.linalg.qr(self.L); self.R, _ = torch.linalg.qr(self.R)
            energy = torch.zeros(self.num_factors, device=device); npr_sum = torch.zeros(self.num_factors, device=device)
            gU, gL, gR = torch.zeros_like(self.U), torch.zeros_like(self.L), torch.zeros_like(self.R); n = 0
            for c in contexts:
                span = make_span(c)

                def objective(u, l, r):
                    H = mixed_hessian(lambda t: span(t, None)[0], l, r)
                    score = u.float() @ H.float()
                    if penalised:
                        cc = mixed_hessian(lambda t: concat_hidden(span(t, None)[1], pr_layers), l, r)
                        npr = participation_ratio(cc) / feature_dim
                        return 0.5 * score.square() - self.penalty_weight * self.penalty_scale * npr, (score, npr)
                    return 0.5 * score.square(), (score, torch.zeros((), device=device))

                grads, (score, npr) = vmap(grad(objective, argnums=(0, 1, 2), has_aux=True), in_dims=(1, 1, 1),
                                           out_dims=((1, 1, 1), (0, 0)), chunk_size=factor_batch)(self.U, self.L, self.R)
                with torch.no_grad():
                    energy += score.square(); npr_sum += npr; gU += grads[0]; gL += grads[1]; gR += grads[2]; n += 1
            with torch.no_grad():
                gU /= n; gL /= n; gR /= n
                self.U = F.normalize(beta * gU + (1 - beta) * self.U, dim=0)
                self.L = F.normalize(beta * gL + (1 - beta) * self.L, dim=0)
                self.R = F.normalize(beta * gR + (1 - beta) * self.R, dim=0)
                mean_energy, mean_pr = energy / n, npr_sum / n
                self.score_energy_values.append(float(mean_energy.sum())); self.normalized_pr_values.append(float(mean_pr.mean()))
                self.objective_values.append(float(0.5 * mean_energy.sum() - self.penalty_weight * self.penalty_scale * mean_pr.sum()))
        return self.U, self.L, self.R


def evaluate(dic: SpanDCT, make_span, contexts, pr_layers):
    """AJ's `evaluate`: per-context, per-factor energy (u.H)^2 and PR over the AJ range; mean total energy; median PR."""
    energies, prs = [], []
    with torch.no_grad():
        for c in contexts:
            span = make_span(c); e, p = [], []
            for f in range(dic.num_factors):
                l, r, u = dic.L[:, f], dic.R[:, f], dic.U[:, f]
                H = mixed_hessian(lambda t: span(t, None)[0], l, r)
                cc = mixed_hessian(lambda t: concat_hidden(span(t, None)[1], pr_layers), l, r)
                e.append((u.float() @ H.float()).square()); p.append(participation_ratio(cc))
            energies.append(torch.stack(e)); prs.append(torch.stack(p))
    energies, prs = torch.stack(energies), torch.stack(prs)
    return {"factor_context_energy": energies.cpu().tolist(), "mean_total_energy": float(energies.sum(1).mean()),
            "median_participation_ratio": float(prs.median()), "mean_participation_ratio": float(prs.mean()),
            "factor_median_pr": prs.median(0).values.cpu().tolist(), "factor_mean_energy": energies.mean(0).cpu().tolist()}
