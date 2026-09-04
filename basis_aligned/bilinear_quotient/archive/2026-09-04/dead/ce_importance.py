"""PER-ATOM CE IMPORTANCE (follows 760B: coupling degree != loss importance -- so
what DOES carry a layer's loss? is it concentrated on a few load-bearing atoms, or
spread?). Take the faithful independent weight-action SAE of Down_0 (750), and for
each ACTIVE atom knock it out of the reconstruction and measure the CE increase.
Ask: is CE-importance CONCENTRATED (a few atoms carry most of the layer's loss,
most atoms present-but-not-load-bearing = parsimony), and does it track USAGE?
Also test SUPERADDITIVITY: does summing single-atom effects over/under-count the
joint knockout (shared vs independent computation)?

REGISTERED PREDICTIONS:
  (0) SANITY: some atoms have nonzero dCE when knocked;
  (a) CONCENTRATED: the top-10 atoms by dCE carry >= 50% of the summed positive dCE
      (few load-bearing atoms; most active atoms are ~0 = not over-explaining);
  (b) dCE correlates with USAGE (used atoms matter more) but only PARTIALLY (rho in
      0.2..0.8 -- usage is necessary not sufficient); report the sum-of-singles vs
      joint-knockout ratio (superadditivity);
  NULL: knocking never-active atoms gives dCE ~ 0."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ce_importance_results.json'
NFIT = 48; NEVAL = 24; P = 512; K = 32
SUB = {'d0': None}; KNOCK = {'mask': None}


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def hook_d0(mo, i_, o_):
    return o_ if SUB['d0'] is None else SUB['d0'](i_[0].reshape(-1, HID)).reshape(o_.shape).to(o_.dtype)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def ce_on(rows, n):
    s = 0.0; nn = 0
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1)
        s += float(F.nll_loss(lp.reshape(-1, lp.shape[-1]), tgt.reshape(-1)))*idx.shape[0]; nn += idx.shape[0]
    return s/nn


@torch.no_grad()
def capture(rows, n):
    cap = []
    h = m.transformer.h[0].mlp.Down.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4): forward_logits(rows[i:i+4, :257].to(DEV)[:, :-1].contiguous())
    h.remove(); return torch.cat(cap, 0)


def sub_d0(Dm, Em, b):
    def fn(g):
        z = topk(g @ Em.T, K)
        if KNOCK['mask'] is not None: z = z * KNOCK['mask']
        return z @ Dm.T + b
    return fn


def train_indep(Xin, Ytrue, seed=1):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(700):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def gini(x):
    x = np.sort(np.clip(x, 0, None)); n = len(x)
    if x.sum() == 0: return 0.0
    return float((2*np.arange(1, n+1) - n - 1).dot(x) / (n * x.sum()))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL); fit, ev = rows[:NFIT], rows[NFIT:NFIT+NEVAL]
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV)
    h0 = m.transformer.h[0].mlp.Down.register_forward_hook(hook_d0)
    g0 = capture(fit, NFIT); Y0 = g0 @ W0.T
    with torch.enable_grad(): Dm, Em, b = train_indep(g0, Y0)
    SUB['d0'] = sub_d0(Dm, Em, b); KNOCK['mask'] = None

    g0_ev = capture(ev, NEVAL); usage = (topk(g0_ev @ Em.T, K) > 1e-6).float().mean(0)   # (P,)
    active = (usage > 0).nonzero(as_tuple=True)[0]
    ce0 = ce_on(ev, NEVAL)
    print(f'active atoms {len(active)}/{P}  ce0 {ce0:.3f}', flush=True)

    # per-atom dCE (knock each active atom individually)
    dce = torch.zeros(P, device=DEV)
    for i in active.tolist():
        mask = torch.ones(P, device=DEV); mask[i] = 0.0; KNOCK['mask'] = mask
        dce[i] = ce_on(ev, NEVAL) - ce0
    KNOCK['mask'] = None
    dce_a = dce[active].cpu().numpy(); pos = np.clip(dce_a, 0, None)
    order = np.argsort(-pos); top10_share = float(pos[order[:10]].sum()/max(pos.sum(), 1e-9))
    gi = gini(pos)
    rho_usage = float(np.corrcoef(usage[active].cpu().numpy(), dce_a)[0, 1])
    print(f'(a) concentration: top-10 share {top10_share:.3f}  Gini {gi:.3f}  '
          f'(rho dCE~usage {rho_usage:.3f})', flush=True)

    # superadditivity: sum of singles vs joint knockout of the top-32 by dCE
    topset = active[torch.tensor(order[:32], device=DEV)]
    mask = torch.ones(P, device=DEV); mask[topset] = 0.0; KNOCK['mask'] = mask
    dce_joint = ce_on(ev, NEVAL) - ce0; KNOCK['mask'] = None
    sum_singles = float(pos[order[:32]].sum())
    superadd = dce_joint/max(sum_singles, 1e-9)
    print(f'(b) top-32 joint dCE {dce_joint:.3f} vs sum-of-singles {sum_singles:.3f}  ratio {superadd:.2f}', flush=True)

    # null: knock never-active atoms
    dead = (usage == 0).nonzero(as_tuple=True)[0]
    if len(dead) >= 8:
        mask = torch.ones(P, device=DEV); mask[dead[:8]] = 0.0; KNOCK['mask'] = mask
        dce_dead = ce_on(ev, NEVAL) - ce0; KNOCK['mask'] = None
    else: dce_dead = 0.0
    print(f'NULL knock 8 never-active atoms: dCE {dce_dead:.4f}', flush=True)
    h0.remove()

    p0 = float(pos.max()) > 1e-3
    pa = top10_share >= 0.5
    pb = 0.2 <= rho_usage <= 0.8
    null_ok = abs(dce_dead) < 0.01
    out = {'n_active': int(len(active)), 'ce0': round(ce0, 4), 'top10_share': round(top10_share, 4),
           'gini': round(gi, 4), 'rho_dce_usage': round(rho_usage, 4), 'top32_joint_dce': round(float(dce_joint), 4),
           'sum_of_singles': round(sum_singles, 4), 'superadd_ratio': round(superadd, 4),
           'null_dead_dce': round(float(dce_dead), 5), 'pred_0': bool(p0), 'pred_a': bool(pa),
           'pred_b': bool(pb), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) CE-importance concentrated (top-10 >=50%): {pa}; (b) tracks usage partially: {pb}; '
          f'NULL dead~0: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
