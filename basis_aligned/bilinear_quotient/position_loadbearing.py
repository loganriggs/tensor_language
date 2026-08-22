"""WHY IS POSITION-ONLY KEEP SO LOAD-BEARING? (§834 follow-up). §834: keeping only the position
subspace at every component recovers 0.595 of the within-class benefit and 0.630 of the class
benefit — far above simultaneous centered-random (−0.10, §833). Is this REAL position information,
or a construction/rank artifact of a rank-32 subspace built from position-conditional means? Clean
control: build a SHUFFLED-position subspace (same rank, same mean-deviation construction, but position
labels randomized) and compare its recovery to the real position subspace. Also break the real
position-only recovery down by position bin (early vs late) — is it concentrated where position is
informative (early) or uniform (structural/operational)?

REGISTERED PREDICTIONS:
  (0) SANITY: real position-only reproduces ~0.60 within / ~0.63 class;
  (a) REAL POSITION load-bearing: shuffled-position subspace recovers FAR LESS than real position
      (real >> shuffled) -> genuine position information, not a rank artifact;
  (b) STRUCTURAL vs INFO: report recovery by position bin — uniform across depth => position is
      structurally/operationally load-bearing; early-concentrated => it predicts early-sequence words;
  NULL: shuffled-position subspace is the matched null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'position_loadbearing_results.json'
NEVAL = 200; MINCOUNT = 5; RPOS = 32
SUBS = {}; MODE = {'op': None, 'key': 'pos'}


def comp(w, L): return m.transformer.h[L].mlp if w == 'mlp' else m.transformer.h[L].attn


def mk_hook(w, L):
    def hook(mo, i_, o_):
        if MODE['op'] is None: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; sh = y.shape; v = y.reshape(-1, D).float()
        if MODE['op'] == 'ablate': v2 = torch.zeros_like(v)
        else: U = SUBS[(MODE['key'], w, L)]; v2 = (v @ U) @ U.T
        yn = v2.reshape(sh).to(y.dtype)
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def ce_by_bin(rows, bins):
    """total CE per position bin."""
    acc = np.zeros(len(bins)); cnt = np.zeros(len(bins))
    for i in range(0, NEVAL, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); logp = F.log_softmax(lg, -1)
        nll = -logp.reshape(-1, logp.shape[-1])[torch.arange(tgt.numel(), device=DEV), tgt.reshape(-1)]
        nll = nll.cpu().numpy(); pos = np.broadcast_to(np.arange(idx.shape[1]), idx.shape).reshape(-1)
        for bi, (lo, hi) in enumerate(bins):
            mk = (pos >= lo) & (pos < hi)
            if mk.any(): acc[bi] += nll[mk].sum(); cnt[bi] += mk.sum()
    return acc/np.clip(cnt, 1, None)


@torch.no_grad()
def capture(rows, w, L):
    cap = []; pos = []
    def h(mo, i_, o_): cap.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = comp(w, L).register_forward_hook(h)
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx)
        pos.append(np.broadcast_to(np.arange(idx.shape[1]), idx.shape).reshape(-1))
    hh.remove(); return torch.cat(cap, 0), np.concatenate(pos)


def mean_subspace(O, labels, r):
    g = O.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < MINCOUNT: continue
        rows.append(O[mk].mean(0) - g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0) * torch.tensor(wt, device=O.device, dtype=O.dtype)[:, None]
    k = min(r, M.shape[0])
    return torch.linalg.svd(M, full_matrices=False)[2][:k].T.contiguous()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    BINS = [(0, 8), (8, 32), (32, 96), (96, 256)]
    comps = [(w, L) for L in range(18) for w in ('attn', 'mlp')]
    MODE['op'] = None; rng = np.random.RandomState(0)
    for w, L in comps:
        O, pos = capture(rows, w, L)
        SUBS[('pos', w, L)] = mean_subspace(O, pos.astype(np.int64), RPOS)
        shuf = pos.copy(); rng.shuffle(shuf)
        SUBS[('shuf', w, L)] = mean_subspace(O, shuf.astype(np.int64), RPOS)
    hooks = [comp(w, L).register_forward_hook(mk_hook(w, L)) for w, L in comps]
    MODE['op'] = None; full = ce_by_bin(rows, BINS)
    MODE['op'] = 'ablate'; abl = ce_by_bin(rows, BINS)
    MODE['op'] = 'keep'; MODE['key'] = 'pos'; keep_pos = ce_by_bin(rows, BINS)
    MODE['op'] = 'keep'; MODE['key'] = 'shuf'; keep_shuf = ce_by_bin(rows, BINS); MODE['op'] = None
    for h in hooks: h.remove()
    ben = abl - full
    rec_pos = (abl - keep_pos)/np.clip(ben, 1e-6, None); rec_shuf = (abl - keep_shuf)/np.clip(ben, 1e-6, None)
    per_bin = []
    for bi, (lo, hi) in enumerate(BINS):
        per_bin.append({'pos_range': [lo, hi], 'rec_position': round(float(rec_pos[bi]), 4), 'rec_shuffled': round(float(rec_shuf[bi]), 4)})
        print(f'pos [{lo},{hi}): position-keep recovers {rec_pos[bi]:.3f} | shuffled-null {rec_shuf[bi]:.3f}', flush=True)
    tot_pos = float((abl.sum()-keep_pos.sum())/max(ben.sum(), 1e-6)); tot_shuf = float((abl.sum()-keep_shuf.sum())/max(ben.sum(), 1e-6))
    out = {'overall_rec_position': round(tot_pos, 4), 'overall_rec_shuffled': round(tot_shuf, 4),
           'per_bin': per_bin, 'position_over_shuffled': round(tot_pos - tot_shuf, 4),
           'pred_a_real_position': bool(tot_pos - tot_shuf > 0.2),
           'uniform_across_depth': bool(max(rec_pos) - min(rec_pos) < 0.25), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\noverall: position-keep {tot_pos:.3f} vs shuffled-position {tot_shuf:.3f} (real-position gain {tot_pos-tot_shuf:+.3f})', flush=True)
    print(f'(a) real position load-bearing (>> shuffled): {out["pred_a_real_position"]} | uniform across depth: {out["uniform_across_depth"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
