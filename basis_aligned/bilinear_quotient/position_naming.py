"""WHAT DOES THE POSITION VARIABLE ENCODE, concretely? (§825 symmetric capstone — class was
named, position is still just 'coarse early/late'). Take the position-conditional-mean mlp0
outputs (per absolute position 0..255), SVD, and characterize the top position directions: is
dir0 a smooth monotonic early->late ramp? are higher dirs curvature, or specific landmarks
(position 0 / very early)? Report each top direction's loading-vs-position curve and its
correlation with (i) linear position, (ii) a position-0 indicator, (iii) log-position.

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-position labels give a flat/incoherent leading direction (control);
  (a) the position variable is LOW-dim (eff ~2-4) and its top direction is smooth & monotonic
      in position (|corr with linear or log position| high); report whether a distinct
      position-0/very-early LANDMARK direction exists;
  (b) report eff-num position dims and the per-direction correlations + curve."""
import json, time, sys, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'position_naming_results.json'; FIG = PT + 'position_naming.png'
NEVAL = 400; SEQ = 256; NDIR = 5; LAYER = 0


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture(rows):
    cap = []; pos = []
    mod = m.transformer.h[LAYER].mlp
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = mod.register_forward_hook(h)
    for i in range(0, rows.shape[0], 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        T = idx.shape[1]; pos.append(np.broadcast_to(np.arange(T), idx.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(pos)


def pos_dirs(O, posl, g):
    ps = sorted(np.unique(posl)); rows = []; kept = []
    for p in ps:
        mk = posl == p
        if mk.sum() < 3: continue
        rows.append(O[mk].mean(0) - g[0]); kept.append(int(p))
    M = torch.stack(rows, 0)                            # (P, D) per-position mean deviation
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    load = M @ Vh.T                                     # (P, K) loading of each position on each dir
    return np.array(kept), load, S


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    O, posl = capture(rows); g = O.mean(0, keepdim=True)
    kept, load, S = pos_dirs(O, posl, g)
    eff = float((S.sum()**2)/(S**2).sum())
    p = kept.astype(np.float64)
    linp = (p - p.mean()); logp = np.log(p + 1); logp = logp - logp.mean(); is0 = (p <= 2).astype(np.float64)
    dirs = []
    for k in range(min(NDIR, load.shape[1])):
        c = load[:, k].cpu().numpy()
        cr_lin = abs(float(np.corrcoef(c, linp)[0, 1])); cr_log = abs(float(np.corrcoef(c, logp)[0, 1])); cr_0 = abs(float(np.corrcoef(c, is0)[0, 1]))
        dirs.append({'k': k, 'sv': round(float(S[k]), 1), 'corr_linear': round(cr_lin, 3),
                     'corr_logpos': round(cr_log, 3), 'corr_pos0': round(cr_0, 3)})
        print(f"dir {k} (sv {float(S[k]):.0f}): |corr| linear {cr_lin:.2f} logpos {cr_log:.2f} pos0 {cr_0:.2f}", flush=True)
    # control
    rng = np.random.RandomState(0); sh = posl.copy(); rng.shuffle(sh)
    _, load2, S2 = pos_dirs(O, sh, g); eff2 = float((S2.sum()**2)/(S2**2).sum())
    c2 = load2[:, 0].cpu().numpy(); ctrl_lin = abs(float(np.corrcoef(c2, linp)[0, 1]))
    # figure: top-3 direction curves vs position
    fig, ax = plt.subplots(figsize=(10, 5)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    cols = ['#3987e5', '#b5852a', '#4a9e6f']
    for k in range(3):
        cvals = load[:, k].cpu().numpy(); cvals = cvals / (np.abs(cvals).max() + 1e-9)
        ax.plot(kept, cvals, color=cols[k], lw=2, label=f'position dir {k} (sv {float(S[k]):.0f})')
    ax.set_xlabel('absolute position', fontsize=10.5, color=INK); ax.set_ylabel('direction loading (normalized)', fontsize=10.5, color=INK)
    ax.set_title('What the position variable encodes: loading of top directions vs position', fontsize=12.5, color=INK, pad=10)
    ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5); ax.tick_params(colors=SECONDARY)
    fig.tight_layout(); fig.savefig(FIG, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    out = {'eff_num_pos_dirs': round(eff, 2), 'eff_shuffled_control': round(eff2, 2),
           'shuffled_dir0_corr_linear': round(ctrl_lin, 3), 'directions': dirs, 'n_positions': len(kept), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\neff # position dims {eff:.2f} (shuffled control {eff2:.2f}); shuffled dir0 corr-linear {ctrl_lin:.2f}", flush=True)
    print(f"wrote {OUT} + {FIG} ({out['runtime_s']:.0f}s)")


if __name__ == '__main__':
    main()
