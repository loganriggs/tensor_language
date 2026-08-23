# pattern_width_by_depth: does the log-local pattern law (§1162) govern heads at depth?
#
# §1162: head 2.5's pattern argmax closes monotonically with front-window width (0.81 at
# W=64) — the front writes a ~64-local identity code even though its loss is 4-local.
# This sweep asks whether OTHER catalogued heads' patterns are window-foldable the same way:
#   3.5  (early-induction trigger, blocks 0-2; ladder code per writeups 397-399)
#   5.5  (the deep induction head, blocks 0-4)
#   8.4  (diffuse deep trigger, blocks 0-7; writeup 398)
# Method identical to m1_width.py: x_hat from weights-only window forward (last-W tokens,
# clamped at 0 — position 0's window IS its true prefix), head pattern from x_hat vs real,
# argmax hit at positions q>=128, W ∈ {8, 32, 128}. Null floor from §1161: 0.18 (2.5);
# recomputed here per head via shuffled-row x_hat at W=32.
#
# Registered predictions:
#   pred_a 3.5 LOG-LOCAL TOO: monotone in W and >= 0.75 at W=128.
#   pred_b 5.5 FOLDS EASILY AT SMALL W: >= 0.7 at W=8 — its dominant read is the position-0
#          constant (sink §429-432/§1089), and position 0's window is exact; the query side
#          is content-insensitive. Alternative (low at all W): sink query needs global state.
#   pred_c 8.4 INTERMEDIATE AND SHALLOW: hit(W=128) − hit(W=8) < the same span for 3.5
#          (diffuse channel per writeup 398 — width helps less).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pattern_width_by_depth_results.json'
NR = 24; WS = [8, 32, 128]
HEADS = [((3, 5), 3), ((5, 5), 5), ((8, 4), 8)]   # (head, nblocks for window forward)
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
def head_pattern_from(X, li, hd, cos, sin):
    at = m.transformer.h[li].attn
    B = X.shape[0]
    qf = F.rms_norm(at.c_q(X).view(B, T, 9, 128), (128,))[:, :, hd]
    kf = F.rms_norm(at.c_k(X).view(B, T, 9, 128), (128,))[:, :, hd]
    q2 = F.rms_norm(at.c_q2(X).view(B, T, 9, 128), (128,))[:, :, hd]
    k2 = F.rms_norm(at.c_k2(X).view(B, T, 9, 128), (128,))[:, :, hd]
    qf = are(qf[:, :, None], cos, sin)[:, :, 0]; kf = are(kf[:, :, None], cos, sin)[:, :, 0]
    q2 = are(q2[:, :, None], cos, sin)[:, :, 0]; k2 = are(k2[:, :, None], cos, sin)[:, :, 0]
    pat = (torch.einsum('bqd,bkd->bqk', qf.float(), kf.float())
           * torch.einsum('bqd,bkd->bqk', q2.float(), k2.float()))
    return pat * torch.tril(torch.ones(T, T, device=DEV))


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NR)[:, :T + 1].contiguous()
    res = {}
    for (li, hd), nb in HEADS:
        for w in WS:
            res[f'{li}.{hd}_W{w}'] = {'hit': 0, 'n': 0}
        res[f'{li}.{hd}_null'] = {'hit': 0, 'n': 0}
    cap = {}
    def mk(li):
        def pre(mo_, args): cap[li] = args[0]
        return pre
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous()
        hs = [m.transformer.h[li].attn.register_forward_pre_hook(mk(li)) for (li, _), _ in HEADS]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        for h in hs: h.remove()
        at0 = m.transformer.h[3].attn
        cos, sin = at0.rotary(at0.c_q(cap[3]).view(4, T, 9, 128))
        sidx = idx[torch.randperm(4, device=DEV)]
        for (li, hd), nb in HEADS:
            rp = head_pattern_from(cap[li], li, hd, cos, sin)
            arms = [(f'{li}.{hd}_W{w}', idx, w) for w in WS] + [(f'{li}.{hd}_null', sidx, 32)]
            for tag, toks, w in arms:
                xh = window_resid(toks, w, nb)
                fp = head_pattern_from(xh, li, hd, cos, sin)
                st = res[tag]
                for b in range(4):
                    for q in range(128, T, 3):
                        kr = int(rp[b, q, :q].abs().argmax()); kf = int(fp[b, q, :q].abs().argmax())
                        st['hit'] += int(kr == kf); st['n'] += 1
    H = {k: round(v['hit'] / max(v['n'], 1), 4) for k, v in res.items()}
    span35 = H['3.5_W128'] - H['3.5_W8']; span84 = H['8.4_W128'] - H['8.4_W8']
    out = {'n_rows': NR, 'hits': H,
           'pred_a_35_loglocal': bool(H['3.5_W8'] < H['3.5_W32'] < H['3.5_W128'] and H['3.5_W128'] >= 0.75),
           'pred_b_55_folds_small': bool(H['5.5_W8'] >= 0.7),
           'pred_c_84_shallow': bool(span84 < span35),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for (li, hd), _ in HEADS:
        print(f"{li}.{hd}: " + " ".join(f"W{w}:{H[f'{li}.{hd}_W{w}']}" for w in WS) +
              f" null:{H[f'{li}.{hd}_null']}", flush=True)
    print(f"pred_a 3.5 loglocal {out['pred_a_35_loglocal']} | pred_b 5.5 small-W {out['pred_b_55_folds_small']} | pred_c 8.4 shallow {out['pred_c_84_shallow']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
