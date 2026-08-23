# pattern_fold_map: the full 162-head window-foldability map.
#
# §1162-63: every head examined (2.5, 3.5, 5.5, 8.4) has an attention pattern computable
# from weights applied to a bounded trailing window (hit 0.78-0.91 @ W=128 vs ~0.25
# positional floor). If that holds model-wide, the model's attention SELECTION machinery is
# n-gram-computable with n≈128 — a strong global interpretability statement, and the
# pattern-side complement to the loss-side map (writeup 482: 4-token windows cost ≤0.086
# everywhere except the L5 sink fetch).
#
# Method: for each layer L (0-17), x_hat(L) = weights-only window forward through blocks
# 0..L-1 on each position's last-W tokens (W ∈ {32, 128}); compute all 9 heads' squared-
# attention patterns from x_hat and from the real residuals; argmax hit at q >= 128,
# stride 5. Null: shuffled-row x_hat at W=32 for layers {2, 9, 16} (positional floor by
# depth band). L0's x_hat is exact by construction (sanity: hit = 1.0).
#
# Registered predictions:
#   pred_a GLOBAL LAW: every layer's MEAN hit @W128 >= 0.5.
#   pred_b NON-MONOTONE DEPTH PROFILE: front (L0-3) mean > middle (L8-12) mean; and
#          readout band (L15-17) mean > middle mean (kernel/recency dominance recovers
#          foldability after the state-dependent dip).
#   pred_c SINK AT CEILING: head 5.7 >= 0.9 at BOTH widths (constant position-0 fetch).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pattern_fold_map_results.json'
NR = 16; WS = [32, 128]; NULL_LAYERS = [2, 9, 16]
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def window_resid(tokens, W, nblocks):
    B, Tn = tokens.shape
    idx = torch.arange(Tn, device=DEV)
    win = torch.stack([tokens[:, (idx + o).clamp_min(0)] for o in range(-(W - 1), 1)], -1)
    flat = win.reshape(B * Tn, W)
    outs = []
    step = max(128, 4096 // W)
    for i in range(0, flat.shape[0], step):
        wb = flat[i:i + step]
        x = F.rms_norm(m.transformer.wte(wb), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h[:nblocks]:
            x, v1 = blk(x, v1, x0)
        outs.append(x[:, -1].detach())
    return torch.cat(outs, 0).reshape(B, Tn, D)


@torch.no_grad()
def all_head_patterns(X, li, cos, sin):
    at = m.transformer.h[li].attn
    B = X.shape[0]
    qf = F.rms_norm(at.c_q(X).view(B, T, 9, 128), (128,))
    kf = F.rms_norm(at.c_k(X).view(B, T, 9, 128), (128,))
    q2 = F.rms_norm(at.c_q2(X).view(B, T, 9, 128), (128,))
    k2 = F.rms_norm(at.c_k2(X).view(B, T, 9, 128), (128,))
    qf = are(qf, cos, sin); kf = are(kf, cos, sin)
    q2 = are(q2, cos, sin); k2 = are(k2, cos, sin)
    pat = (torch.einsum('bqhd,bkhd->bhqk', qf.float(), kf.float())
           * torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()))
    return pat * torch.tril(torch.ones(T, T, device=DEV))


@torch.no_grad()
def hits(rp, fp):
    # rp, fp: (B, 9, T, T); returns per-head hit counts over q>=128 stride 5
    h = torch.zeros(9, device=DEV); n = 0
    for q in range(128, T, 5):
        kr = rp[:, :, q, :q].abs().argmax(-1); kf = fp[:, :, q, :q].abs().argmax(-1)
        h += (kr == kf).float().sum(0); n += rp.shape[0]
    return h.cpu(), n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    acc = {f'L{L}_W{w}': [torch.zeros(9), 0] for L in range(18) for w in WS}
    nul = {f'L{L}_null': [torch.zeros(9), 0] for L in NULL_LAYERS}
    cap = {}
    def mk(li):
        def pre(mo_, args): cap[li] = args[0]
        return pre
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous()
        hs = [m.transformer.h[li].attn.register_forward_pre_hook(mk(li)) for li in range(18)]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        for h in hs: h.remove()
        at0 = m.transformer.h[0].attn
        cos, sin = at0.rotary(at0.c_q(cap[0]).view(4, T, 9, 128))
        sidx = idx[torch.randperm(4, device=DEV)]
        for L in range(18):
            rp = all_head_patterns(cap[L], L, cos, sin)
            for w in WS:
                xh = window_resid(idx, w, L)
                fp = all_head_patterns(xh, L, cos, sin)
                hv, n = hits(rp, fp)
                acc[f'L{L}_W{w}'][0] += hv; acc[f'L{L}_W{w}'][1] += n
            if L in NULL_LAYERS:
                xh = window_resid(sidx, 32, L)
                fp = all_head_patterns(xh, L, cos, sin)
                hv, n = hits(rp, fp)
                nul[f'L{L}_null'][0] += hv; nul[f'L{L}_null'][1] += n
        print(f"batch {i // 4 + 1}/{NR // 4} done {round(time.time() - t0)}s", flush=True)

    table = {}
    for L in range(18):
        for w in WS:
            hv, n = acc[f'L{L}_W{w}']
            table[f'L{L}_W{w}'] = [round(float(v) / max(n, 1), 3) for v in hv]
    nulls = {k: [round(float(v) / max(n, 1), 3) for v in hv] for k, (hv, n) in nul.items()}
    mean128 = {L: sum(table[f'L{L}_W128']) / 9 for L in range(18)}
    front = sum(mean128[L] for L in range(4)) / 4
    mid = sum(mean128[L] for L in range(8, 13)) / 5
    late = sum(mean128[L] for L in range(15, 18)) / 3
    sink57 = table['L5_W32'][7], table['L5_W128'][7]
    out = {'n_rows': NR, 'per_head': table, 'nulls_W32_shuffled': nulls,
           'layer_mean_W128': {str(L): round(mean128[L], 3) for L in range(18)},
           'band_means_W128': {'front_L0_3': round(front, 3), 'middle_L8_12': round(mid, 3),
                               'late_L15_17': round(late, 3)},
           'sink_5_7': {'W32': sink57[0], 'W128': sink57[1]},
           'pred_a_global_law': bool(all(mean128[L] >= 0.5 for L in range(18))),
           'pred_b_nonmonotone': bool(front > mid and late > mid),
           'pred_c_sink_ceiling': bool(sink57[0] >= 0.9 and sink57[1] >= 0.9),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for L in range(18):
        print(f"L{L:>2} W128 mean {mean128[L]:.3f} | heads {table[f'L{L}_W128']}", flush=True)
    print(f"bands front {out['band_means_W128']['front_L0_3']} mid {out['band_means_W128']['middle_L8_12']} late {out['band_means_W128']['late_L15_17']} | sink 5.7 {out['sink_5_7']}")
    print(f"pred_a global {out['pred_a_global_law']} | pred_b nonmonotone {out['pred_b_nonmonotone']} | pred_c sink {out['pred_c_sink_ceiling']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
