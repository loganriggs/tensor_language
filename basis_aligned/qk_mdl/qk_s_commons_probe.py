"""Content analyses for the w1152 commons192 checkpoint (promised to Logan):
what IS the commons carrying?

1. commons_ledger (qk_e14, width-generic): per-module commons write norms
   (data) + per-consumer commons read norms (weights).
2. var_light_probe (qk_e14): R2 consumption graph + wiring Spearman.
3. commons_stats (new): capture the residual at each slot_norm call site
   (calls are counted on a probe pass; site l*k = entry of layer l), then
   per layer for the 192 commons dims:
     - effective rank (participation ratio of singular values),
     - token-determined R2: fraction of commons variance explained by the
       current token identity (per-token means over tokens with >=10 occs),
     - slot overlap: R2 of least-squares prediction of commons content from
       each single slot's content (is the commons a copy of some slot?) and
       from all slots jointly (is it linearly redundant with the slots?).

Probe eval rows = substitute cooc corpus (same caveat as every w1152 wiring
probe on this box); commons_stats rows = scale held (never trained).
Merges into qk_s_w1152_commons192.json. GPU 1, gated behind the share chain.
"""
import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch

import qk_s_gate_run as G
import qk_w1152_train as W2
import qk_e_common as E
from qk_e_common import Q, DEPTH
import qk_s_e14c_run as C14
import qk_e14_slotcap_run as E14

STEM = 'qk_s_w1152_commons192'
JP = os.path.join(G.OUT_DIR, f'{STEM}.json')
NSUB = 20000                       # rows used for svd/lstsq per layer


def load_model():
    m = C14.make_commons192()
    ck = torch.load(os.path.join(G.OUT_DIR, f'{STEM}.pt'),
                    map_location='cuda', weights_only=False)
    m.load_state_dict(ck['state_dict'])
    return m.eval().float()


@torch.no_grad()
def commons_stats(m, held, n_seq=400, occ_min=10):
    cdim, a = m.commons, Q.D - m.commons
    orig = type(m).slot_norm

    # probe pass: how many slot_norm calls per forward?
    calls = {'n': 0}

    def counting(self, x):
        calls['n'] += 1
        return orig(self, x)

    type(m).slot_norm = counting
    try:
        m(held[:2, :Q.T])
    finally:
        type(m).slot_norm = orig
    npass = calls['n']
    assert npass % DEPTH == 0, npass
    k = npass // DEPTH             # calls per layer (attn entry, mlp entry, ...)
    sites = {l * k: l for l in range(DEPTH)}   # layer-entry call indices
    print(f"slot_norm: {npass} calls/pass = {k} per layer; capturing layer "
          f"entries {sorted(sites)}", flush=True)

    rows = {l: [] for l in range(DEPTH)}
    toks = []
    state = {'i': 0}

    def capture(self, x):
        li = sites.get(state['i'])
        if li is not None:
            rows[li].append(x.reshape(-1, Q.D).float().cpu())
        state['i'] += 1
        return orig(self, x)

    type(m).slot_norm = capture
    try:
        for i in range(0, n_seq, 4):
            b = held[i:i + 4, :Q.T]
            toks.append(b.reshape(-1).cpu())
            state['i'] = 0
            m(b)
    finally:
        type(m).slot_norm = orig

    tok = torch.cat(toks).numpy()
    out = {}
    for li in range(DEPTH):
        X = torch.cat(rows[li]).numpy()
        rows[li] = None
        Cm0 = X[:, a:] - X[:, a:].mean(0)
        s = np.linalg.svd(Cm0[:NSUB], compute_uv=False)
        er = float((s ** 2).sum() ** 2 / (s ** 4).sum())
        tv = float((Cm0 ** 2).sum())
        expl = 0.0
        uniq, counts = np.unique(tok, return_counts=True)
        for t in uniq[counts >= occ_min]:
            idx = np.where(tok == t)[0]
            mu = Cm0[idx].mean(0)
            expl += len(idx) * float((mu ** 2).sum())
        r2_tok = expl / tv
        Y = Cm0[:NSUB]
        yv = float((Y ** 2).sum())
        best_slot, best_r2 = -1, 0.0
        for kk in range(len(m.seg_sizes) - 1):
            sa, sb = m.seg_bounds[kk]
            S = np.concatenate([X[:NSUB, sa:sb],
                                np.ones((len(Y), 1))], 1)
            beta, *_ = np.linalg.lstsq(S, Y, rcond=None)
            r2 = 1 - float(((Y - S @ beta) ** 2).sum()) / yv
            if r2 > best_r2:
                best_slot, best_r2 = kk, float(r2)
        Sall = np.concatenate([X[:NSUB, :a], np.ones((len(Y), 1))], 1)
        beta, *_ = np.linalg.lstsq(Sall, Y, rcond=None)
        r2_all = 1 - float(((Y - Sall @ beta) ** 2).sum()) / yv
        nm = f'{"attn" if best_slot % 2 == 0 else "mlp"}{best_slot // 2}'
        out[f'layer{li}'] = {'eff_rank': round(er, 1),
                             'token_R2': round(r2_tok, 4),
                             'best_single_slot': nm,
                             'best_single_slot_R2': round(best_r2, 4),
                             'all_slots_R2': round(float(r2_all), 4)}
        print(f"layer {li}: eff_rank {er:.1f}/{cdim} token_R2 {r2_tok:.3f} "
              f"best slot {nm} ({best_r2:.3f}) all-slots {r2_all:.3f}",
              flush=True)
        del X, Cm0
    out['calls_per_layer'] = k
    out['commons_dims'] = cdim
    return out


def main():
    W2.patch_width(G.WIDTH)
    G.setup_data()
    E.WIDTH, E.SUB = G.WIDTH, G.WIDTH // E.NGROUP
    out = G.loadj(JP)
    m = load_model()
    held = torch.from_numpy(
        np.asarray(np.load(os.path.join(G.QK, 'corpus_fresh/fresh34k.npy'),
                           mmap_mode='r')[33000:33400]).astype(np.int64)).cuda()
    if 'commons_ledger' not in out:
        out['commons_ledger'] = E14.commons_ledger(m)
        G.savej(JP, out)
        print('ledger done', flush=True)
    if 'commons_stats' not in out:
        out['commons_stats'] = commons_stats(m, held)
        out['commons_stats']['rows'] = 'fresh34k[33000:33400] (never trained)'
        G.savej(JP, out)
        print('stats done', flush=True)
    if 'light_probe' not in out:
        out['light_probe'] = E14.var_light_probe(m)
        out['light_probe']['caveat'] = ('probe eval rows = substitute cooc '
                                        'corpus, same as all w1152 probes')
        G.savej(JP, out)
    print('commons probes done', flush=True)


if __name__ == '__main__':
    main()
