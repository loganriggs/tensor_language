"""KNOB COMPOSITION -- do bilin18's two demonstrated control knobs act as
INDEPENDENT, composable axes? 691 showed the two mechanisms (embedding-
dominance / gain-controller vs the frequency-calibration direction) are
separate. Control-phase corollary: the TEMPERATURE knob (scale the 8
massive dims by g, 693) and the FREQUENCY-BIAS knob (scale the w_freq
calibration component by alpha, w_freq_steering) should compose
independently -- applying one should not destroy the other's effect.

Test on a 3x3 grid of (g in {0.6,1.0,1.5}) x (alpha in {0.5,1.0,1.75}):
measure output entropy (temperature axis) and top-20 frequent-token mass
(frequency axis). Independence = the temperature knob's entropy effect is
roughly constant across alpha, and the frequency knob's top20-mass effect
is roughly constant across g (no strong interaction).

REGISTERED PREDICTIONS:
  (0) SANITY: (g=1,alpha=1) reproduces baseline; the g-axis moves entropy
      and the alpha-axis moves top20-mass (each knob works, 693 + w_freq);
  (a) INDEPENDENT AXES: the entropy change from g=0.6->1.5 has the same
      SIGN and comparable magnitude at every alpha (ratio of the effect at
      alpha=0.5 vs 1.75 within 2x), AND the top20-mass change from
      alpha=0.5->1.75 has the same sign at every g -- the two knobs do not
      cancel or swap each other;
  (b) report the 3x3 entropy and top20-mass grids;
  NULL/CONTROL: the OFF-DIAGONAL cross-effect is small -- g barely moves
      top20-mass at fixed alpha relative to how alpha moves it (g is a
      temperature axis, not a frequency axis), and alpha barely moves
      entropy relative to how g moves it."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'knob_composition_results.json'
NFRESH = 24
K = 8
GS = [0.6, 1.0, 1.5]
ALPHAS = [0.5, 1.0, 1.75]


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    V = m.lm_head.weight.shape[0]

    # top-20 frequent token ids (by corpus count)
    freq = np.bincount(fresh[:, 1:257].reshape(-1).numpy(), minlength=V).astype(np.float64)
    top20 = torch.tensor(np.argsort(-freq)[:20].tolist(), device=DEV)

    # find massive dims + w_freq direction (cov of mlp17 out with log-freq)
    ss = torch.zeros(D, dtype=torch.float64); n = 0
    mlp17_out = []; logf_cap = []
    cap = {}
    h = m.transformer.h[17].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.__setitem__('o', o_.detach().float()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h):
            x, v1 = blk(x, v1, x0)
        ss += (x.float() ** 2).reshape(-1, D).sum(0).double().cpu(); n += idx.numel()
        mlp17_out.append(cap['o'].reshape(-1, D).cpu())
        lf = torch.tensor(np.log(freq[tgt.reshape(-1).cpu().numpy()] + 1.0))
        logf_cap.append(lf)
    h.remove()
    rms = np.sqrt((ss / n).numpy())
    massive = torch.tensor(np.argsort(-rms)[:K].tolist(), device=DEV)
    O = torch.cat(mlp17_out, 0); lf = torch.cat(logf_cap, 0)
    wf = ((O - O.mean(0)) * (lf - lf.mean())[:, None]).mean(0)
    wf = (wf / wf.norm()).float().to(DEV)

    # forward with both interventions: scale w_freq comp of mlp17 out by alpha,
    # scale massive dims of final residual by g.
    W = {'alpha': 1.0}
    def mlp17_hook(mo, i_, o_):
        of = o_.float()
        comp = (of @ wf)[..., None] * wf
        return (of + (W['alpha'] - 1.0) * comp).to(o_.dtype)
    hh = m.transformer.h[17].mlp.register_forward_hook(mlp17_hook)

    @torch.no_grad()
    def measure(g, alpha):
        W['alpha'] = alpha
        ent_s = 0.0; mass_s = 0.0; n = 0
        for i in range(0, NFRESH, 4):
            bb = fresh[i:i + 4, :257].to(DEV)
            idx = bb[:, :-1].contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for li, blk in enumerate(m.transformer.h):
                x, v1 = blk(x, v1, x0)
            x = x.clone(); x[..., massive] = x[..., massive] * g
            logits = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            p = F.softmax(logits.float(), -1)
            lp = torch.log(p + 1e-12)
            ent = -(p * lp).sum(-1)
            mass = p[..., top20].sum(-1)
            ent_s += float(ent.mean()) * idx.shape[0]
            mass_s += float(mass.mean()) * idx.shape[0]; n += idx.shape[0]
        return ent_s / n, mass_s / n

    ent_grid = {}; mass_grid = {}
    for g in GS:
        for a in ALPHAS:
            e, mss = measure(g, a)
            ent_grid[f'{g}_{a}'] = round(e, 4); mass_grid[f'{g}_{a}'] = round(mss, 4)
            print(f'g={g} alpha={a}: entropy {e:.4f}  top20mass {mss:.4f}', flush=True)
    hh.remove()

    # entropy effect of g (0.6->1.5) at each alpha
    ent_eff = {a: ent_grid[f'1.5_{a}'] - ent_grid[f'0.6_{a}'] for a in ALPHAS}
    # mass effect of alpha (0.5->1.75) at each g
    mass_eff = {g: mass_grid[f'{g}_1.75'] - mass_grid[f'{g}_0.5'] for g in GS}
    print(f'\nentropy effect of g (0.6->1.5) by alpha: {ent_eff}', flush=True)
    print(f'top20-mass effect of alpha (0.5->1.75) by g: {mass_eff}', flush=True)

    ent_vals = list(ent_eff.values()); mass_vals = list(mass_eff.values())
    ent_same_sign = all(v > 0 for v in ent_vals) or all(v < 0 for v in ent_vals)
    mass_same_sign = all(v > 0 for v in mass_vals) or all(v < 0 for v in mass_vals)
    ent_ratio = max(abs(v) for v in ent_vals) / (min(abs(v) for v in ent_vals) + 1e-9)
    mass_ratio = max(abs(v) for v in mass_vals) / (min(abs(v) for v in mass_vals) + 1e-9)
    pa = ent_same_sign and mass_same_sign and ent_ratio < 2.0
    # cross-effect: does g move mass much vs alpha? does alpha move entropy vs g?
    g_on_mass = abs(mass_grid['1.5_1.0'] - mass_grid['0.6_1.0'])
    a_on_mass = abs(mass_grid['1.0_1.75'] - mass_grid['1.0_0.5'])
    a_on_ent = abs(ent_grid['1.0_1.75'] - ent_grid['1.0_0.5'])
    g_on_ent = abs(ent_grid['1.5_1.0'] - ent_grid['0.6_1.0'])
    null_ok = g_on_mass < a_on_mass and a_on_ent < g_on_ent
    print(f'\n(a) independent axes (same sign, entropy ratio {ent_ratio:.2f}<2): {pa}',
          flush=True)
    print(f'NULL cross-effects small: g-on-mass {g_on_mass:.3f}<a-on-mass {a_on_mass:.3f}, '
          f'a-on-ent {a_on_ent:.3f}<g-on-ent {g_on_ent:.3f}: {null_ok}', flush=True)

    out = {'entropy_grid': ent_grid, 'top20mass_grid': mass_grid,
           'entropy_effect_of_g_by_alpha': {str(k): round(v, 4) for k, v in ent_eff.items()},
           'mass_effect_of_alpha_by_g': {str(k): round(v, 4) for k, v in mass_eff.items()},
           'pred_a_independent': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
