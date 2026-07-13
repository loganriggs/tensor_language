"""e4: reconciliation with Vaintrob/Mendel/Hanni ("Computation in Superposition").

Their quadratic U-AND (post section 1.5) is a bilinear layer + LINEAR readout --
the same architecture as e3. So the e3 rank bound cannot be dodged by "they have
a nonlinear readout" (that sentence in FINDING 4 is RETRACTED). The actual
discriminator is the ERROR METRIC, which they flag themselves (section 2,
"epsilon-accuracy permits much more superposition than minimising the MSE"):

  - MSE: a square that fires w.p. p is worth ~p*E[x^4]; predict-zero nearly
    nails it, and superposition interference is paid on ~every input. The rank
    bound holds and superposition LOSES under MSE (e3).
  - eps-accuracy: the signal is worth 1 whenever present; interference only
    needs to stay < eps on each individual sparse input. Almost-orthogonal
    readoffs then give m >> d_h computed squares (their construction).

Demonstration on ONE task/architecture (y = x^2, bilinear, d_h=32, m=128):

  hand-coded superposition: unit vectors v_i in R^32 with minimized coherence,
  L[:,i] o R[:,i] = v_i, D = V^T. On 1-active inputs x = v*e_j the readout is
  yhat_i = v^2 * G_ij (G = V^T V): every feature exact on its own diagonal
  (G_ii = 1), worst-case error = max coherence.

  training contrast, SAME 1-active data, only the loss differs:
    MSE  -> rank bound says dedicated-like, ~d_h features
    L8   -> (their section-2 suggestion of high-p losses as eps-surrogates)
            should find all-m superposition

Metrics: n_computed under the eps-criterion (worst error of output i over all
1-active inputs <= 0.25), MSE on 1-active and on p-sparse inputs vs baselines,
and worst 2-active error (honest cross-term caveat: this toy construction is
the ell=1 version; their full r^2-neuron construction controls higher ell).
"""

import json
import sys

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
from common import forward, squares_data

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
M, D_H, P_ACT = 128, 32, 0.05
E_V4 = 0.2  # E[v^4] for v ~ U(-1,1)
EPS_TOL = 0.25
torch.manual_seed(0)

BASELINES = {
    'psparse_predict_zero': P_ACT * E_V4,
    'psparse_dedicated': (1 - D_H / M) * P_ACT * E_V4,
    'psparse_rank_bound': (M - D_H) * (P_ACT * E_V4 - (P_ACT / 3) ** 2) / M,
    'oneactive_predict_zero': E_V4 / M,
    'oneactive_dedicated': (1 - D_H / M) * E_V4 / M,
}


def one_active_data(batch):
    j = torch.randint(0, M, (batch,), device=DEV)
    v = torch.rand(batch, device=DEV) * 2 - 1
    x = torch.zeros(batch, M, device=DEV)
    x[torch.arange(batch, device=DEV), j] = v
    return x, x ** 2


@torch.no_grad()
def eval_mse(p, data, n_batches=32):
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = data()
        tot += ((forward(p, x) - y) ** 2).sum().item()
        n += y.numel()
    return tot / n


@torch.no_grad()
def one_active_metrics(p, tol=EPS_TOL):
    """Worst error of each output over all 1-active inputs (exact via v-grid)."""
    vs = torch.linspace(-1, 1, 21, device=DEV)
    worst = torch.zeros(M, device=DEV)
    for j in range(M):
        x = vs[:, None] * torch.eye(M, device=DEV)[j][None, :]
        y = torch.zeros(len(vs), M, device=DEV)
        y[:, j] = vs ** 2
        err = (forward(p, x) - y).abs().amax(0)
        worst = torch.maximum(worst, err)
    return {'n_computed': int((worst <= tol).sum()),
            'worst_err_median': float(worst.median()),
            'worst_err_max': float(worst.max())}


@torch.no_grad()
def two_active_worst(p, n=50000):
    i = torch.randint(0, M, (n,), device=DEV)
    j = torch.randint(0, M, (n,), device=DEV)
    u = torch.rand(n, device=DEV) * 2 - 1
    w = torch.rand(n, device=DEV) * 2 - 1
    x = torch.zeros(n, M, device=DEV)
    r = torch.arange(n, device=DEV)
    x[r, i] = u
    x[r, j] += w
    return float((forward(p, x) - x ** 2).abs().amax(1).max())


# ---------------------------------------------------------------- constructions

def low_coherence_frame(d, m, steps=4000):
    V = torch.randn(d, m, device=DEV)
    V = V / V.norm(dim=0)
    V.requires_grad_(True)
    opt = torch.optim.Adam([V], lr=3e-3)
    for _ in range(steps):
        Vn = V / V.norm(dim=0)
        G = Vn.T @ Vn
        off = G - torch.eye(m, device=DEV)
        loss = torch.logsumexp(60 * off.abs().flatten(), 0) / 60
        opt.zero_grad(); loss.backward(); opt.step()
    V = (V / V.norm(dim=0)).detach()
    return V, float((V.T @ V - torch.eye(m, device=DEV)).abs().max())


def superposition_handcoded(V):
    return {'L': V.sign() * V.abs().sqrt(), 'R': V.abs().sqrt(),
            'D': V.T.contiguous()}


def dedicated_handcoded():
    L = torch.zeros(D_H, M, device=DEV)
    L[torch.arange(D_H), torch.arange(D_H)] = 1.0
    D = torch.zeros(M, D_H, device=DEV)
    D[torch.arange(D_H), torch.arange(D_H)] = 1.0
    return {'L': L, 'R': L.clone(), 'D': D}


# ---------------------------------------------------------------- training

def train_loss(p, steps, lr, loss_kind):
    for k in p:
        p[k].requires_grad_(True)
    opt = torch.optim.Adam(list(p.values()), lr=lr)
    for _ in range(steps):
        x, y = one_active_data(2048)
        err = forward(p, x) - y
        loss = (err ** 2).mean() if loss_kind == 'mse' else (err ** 8).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    for k in p:
        p[k].requires_grad_(False)
    return p


def scratch(seed):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    def mat(a, b):
        return (torch.randn(a, b, generator=g) / b ** 0.5).to(DEV)
    return {'L': mat(D_H, M), 'R': mat(D_H, M), 'D': mat(M, D_H)}


# ---------------------------------------------------------------- run

V, coherence = low_coherence_frame(D_H, M)
welch = ((M - D_H) / (D_H * (M - 1))) ** 0.5
print(f'coherence of optimized frame: {coherence:.3f} (Welch bound {welch:.3f})')

arms = {
    'dedicated_handcoded': dedicated_handcoded(),
    'superposition_handcoded': superposition_handcoded(V),
    'scratch_mse': train_loss(scratch(0), 20000, 3e-3, 'mse'),
    'scratch_L8': train_loss(scratch(0), 20000, 3e-3, 'l8'),
    'superpos_then_mse_ft': train_loss(
        {k: v.clone() for k, v in superposition_handcoded(V).items()}, 10000, 1e-3, 'mse'),
    'superpos_then_L8_ft': train_loss(
        {k: v.clone() for k, v in superposition_handcoded(V).items()}, 10000, 1e-3, 'l8'),
}
arms['scratch_mse'] = train_loss(arms['scratch_mse'], 5000, 3e-4, 'mse')
arms['scratch_L8'] = train_loss(arms['scratch_L8'], 5000, 3e-4, 'l8')

psparse = lambda: squares_data(2048, M, P_ACT, DEV)
results = {'m': M, 'd_h': D_H, 'p_active': P_ACT, 'eps_tol': EPS_TOL,
           'coherence': coherence, 'welch_bound': welch,
           'baselines': BASELINES, 'arms': {}}
print(f"\n{'arm':26s} {'n_comp':>8s} {'worst1(med/max)':>16s} {'mse_1act':>9s} "
      f"{'mse_psparse':>11s} {'worst2':>7s}")
for name, p in arms.items():
    row = {'mse_1active': eval_mse(p, lambda: one_active_data(2048)),
           'mse_psparse': eval_mse(p, psparse),
           **one_active_metrics(p), 'worst_2active': two_active_worst(p)}
    results['arms'][name] = row
    print(f"{name:26s} {row['n_computed']:4d}/{M} "
          f"{row['worst_err_median']:8.3f}/{row['worst_err_max']:5.3f} "
          f"{row['mse_1active']:9.2e} {row['mse_psparse']:11.2e} "
          f"{row['worst_2active']:7.3f}")
print('\nbaselines:', {k: f'{v:.2e}' for k, v in BASELINES.items()})

with open('/workspace/tensor_language/basis_aligned/e4_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
torch.save({k: {kk: vv.cpu() for kk, vv in v.items()} for k, v in arms.items()},
           '/workspace/tensor_language/basis_aligned/e4_states.pt')
print('saved e4_results.json')
