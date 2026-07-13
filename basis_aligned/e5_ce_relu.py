"""e5: where does CE sit on the MSE <-> eps-accuracy axis, and does a nonlinear
readout rescue superposition under MSE?

Same architecture and 1-active data as e4 (m=128, d_h=32). New arms:

  ce           bilinear outputs treated as logits -> softmax + CE, task =
               "which feature is active". Bilinear logits for x = v*e_j are
               v^2 * (D v_j): scale-free argmax, so CE only needs column-wise
               diagonal dominance of the rank-d_h matrix C = D V -- a
               threshold criterion, like eps-accuracy. Prediction: ~all m.
  ce_smooth    CE with label smoothing 0.9 (nearly flat targets) -- probes the
               "peakedness" axis: margins should collapse.
  mse_relu     squares task, MSE loss, but readout ReLU(Dh + b) with learned
               bias -- the TMS route: the nonlinearity clips sub-threshold
               interference, so superposition can win under MSE, and the
               1-active MSE can go BELOW the linear-readout rank bound.

Audits applied to every arm (new + the four saved e4 arms):
  eps-audit    worst error of output i over 1-active inputs (v-grid [-1,1])
               <= 0.25  [squares-task arms only]
  cls-audit    feature j computed iff its output strictly wins argmax for all
               1-active inputs with |v| in [0.5, 1]; report count + accuracy
               on random 1-active samples.
"""

import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace/tensor_language/basis_aligned')
from common import forward

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
M, D_H = 128, 32
EPS_TOL = 0.25
ONEACT_DEDICATED_MSE = (1 - D_H / M) * 0.2 / M  # linear-readout baselines from e4
ONEACT_RANK_BOUND = (M - D_H) * 0.2 / M / M     # = dedicated here (1-active Gram is diagonal)
torch.manual_seed(0)


def fwd_lin(p, x):
    return forward(p, x)


def fwd_relu(p, x):
    return torch.relu(forward(p, x) + p['b'])


def one_active_squares(batch):
    j = torch.randint(0, M, (batch,), device=DEV)
    v = torch.rand(batch, device=DEV) * 2 - 1
    x = torch.zeros(batch, M, device=DEV)
    x[torch.arange(batch, device=DEV), j] = v
    return x, x ** 2


def one_active_class(batch):
    """|v| in [0.25, 1] so the class is always inferable."""
    j = torch.randint(0, M, (batch,), device=DEV)
    v = (torch.rand(batch, device=DEV) * 0.75 + 0.25) * \
        (torch.randint(0, 2, (batch,), device=DEV) * 2 - 1)
    x = torch.zeros(batch, M, device=DEV)
    x[torch.arange(batch, device=DEV), j] = v
    return x, j


# ---------------------------------------------------------------- audits

@torch.no_grad()
def eps_audit(p, fwd, tol=EPS_TOL):
    vs = torch.linspace(-1, 1, 21, device=DEV)
    worst = torch.zeros(M, device=DEV)
    for j in range(M):
        x = vs[:, None] * torch.eye(M, device=DEV)[j][None, :]
        y = torch.zeros(len(vs), M, device=DEV)
        y[:, j] = vs ** 2
        worst = torch.maximum(worst, (fwd(p, x) - y).abs().amax(0))
    return {'n_eps': int((worst <= tol).sum()),
            'worst1_med': float(worst.median())}


@torch.no_grad()
def cls_audit(p, fwd):
    vs = torch.cat([torch.linspace(-1, -0.5, 6), torch.linspace(0.5, 1, 6)]).to(DEV)
    margin = torch.empty(M, device=DEV)
    for j in range(M):
        x = vs[:, None] * torch.eye(M, device=DEV)[j][None, :]
        out = fwd(p, x)
        own = out[:, j].clone()
        out[:, j] = -float('inf')
        margin[j] = (own - out.amax(1)).min()
    x, j = one_active_class(20000)
    acc = float((fwd(p, x).argmax(1) == j).float().mean())
    return {'n_cls': int((margin > 0).sum()), 'acc': acc}


@torch.no_grad()
def mse_1active(p, fwd, n_batches=32):
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = one_active_squares(2048)
        tot += ((fwd(p, x) - y) ** 2).sum().item()
        n += y.numel()
    return tot / n


# ---------------------------------------------------------------- training

def scratch(seed, relu=False):
    g = torch.Generator(device='cpu'); g.manual_seed(seed)
    def mat(a, b):
        return (torch.randn(a, b, generator=g) / b ** 0.5).to(DEV)
    p = {'L': mat(D_H, M), 'R': mat(D_H, M), 'D': mat(M, D_H)}
    if relu:
        p['b'] = torch.zeros(M, device=DEV)
    return p


def train(p, fwd, loss_fn, steps, lr):
    for k in p:
        p[k].requires_grad_(True)
    opt = torch.optim.Adam(list(p.values()), lr=lr)
    for _ in range(steps):
        loss = loss_fn(p, fwd)
        opt.zero_grad(); loss.backward(); opt.step()
    for k in p:
        p[k].requires_grad_(False)
    return p


def loss_mse_squares(p, fwd):
    x, y = one_active_squares(2048)
    return ((fwd(p, x) - y) ** 2).mean()


def loss_ce(smoothing):
    def f(p, fwd):
        x, j = one_active_class(2048)
        return F.cross_entropy(fwd(p, x), j, label_smoothing=smoothing)
    return f


# ---------------------------------------------------------------- run

e4 = torch.load('/workspace/tensor_language/basis_aligned/e4_states.pt')
arms = {name: ({k: v.to(DEV) for k, v in e4[name].items()}, fwd_lin, 'squares')
        for name in ['dedicated_handcoded', 'superposition_handcoded',
                     'scratch_mse', 'scratch_L8']}

print('training ce...')
p = train(scratch(1), fwd_lin, loss_ce(0.0), 20000, 3e-3)
arms['ce'] = (train(p, fwd_lin, loss_ce(0.0), 5000, 3e-4), fwd_lin, 'class')
print('training ce_smooth09...')
p = train(scratch(1), fwd_lin, loss_ce(0.9), 20000, 3e-3)
arms['ce_smooth09'] = (train(p, fwd_lin, loss_ce(0.9), 5000, 3e-4), fwd_lin, 'class')
print('training mse_relu...')
p = train(scratch(2, relu=True), fwd_relu, loss_mse_squares, 20000, 3e-3)
arms['mse_relu'] = (train(p, fwd_relu, loss_mse_squares, 5000, 3e-4), fwd_relu, 'squares')

results = {'m': M, 'd_h': D_H, 'eps_tol': EPS_TOL,
           'oneactive_dedicated_mse': ONEACT_DEDICATED_MSE, 'arms': {}}
print(f"\n{'arm':26s} {'task':>8s} {'n_eps':>7s} {'worst1':>7s} {'n_cls':>7s} "
      f"{'acc':>6s} {'mse_1act':>9s}")
for name, (p, fwd, task) in arms.items():
    row = {'task': task, **cls_audit(p, fwd)}
    if task == 'squares':
        row.update(eps_audit(p, fwd))
        row['mse_1active'] = mse_1active(p, fwd)
    results['arms'][name] = row
    print(f"{name:26s} {task:>8s} "
          f"{row.get('n_eps', '—'):>4}/{M if task == 'squares' else '—':<3} "
          f"{row.get('worst1_med', float('nan')):7.3f} {row['n_cls']:4d}/{M} "
          f"{row['acc']:6.1%} "
          + (f"{row['mse_1active']:9.2e}" if task == 'squares' else '        —'))
print(f"\nlinear-readout dedicated/rank-bound MSE (1-active): {ONEACT_DEDICATED_MSE:.2e}")

with open('/workspace/tensor_language/basis_aligned/e5_results.json', 'w') as fh:
    json.dump(results, fh, indent=2)
torch.save({k: {kk: vv.cpu() for kk, vv in v[0].items()}
            for k, v in arms.items() if k in ('ce', 'ce_smooth09', 'mse_relu')},
           '/workspace/tensor_language/basis_aligned/e5_states.pt')
print('saved e5_results.json')
