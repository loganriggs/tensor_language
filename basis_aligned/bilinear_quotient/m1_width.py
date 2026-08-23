# m1_width: how wide a context window does the front residual need before layer-2's
# pattern closes? (§1161 registered follow-up.)
#
# §1161: the loss-certified 4-gram window forward (blocks 0-1) predicts head 2.5's pattern
# argmax at only 0.577 (48% of closable range) — loss-locality (writeups 480-481: W=4 costs
# +0.014 nats) is NOT pattern-locality. Two hypotheses: (i) pattern-locality is simply WIDER
# but bounded (log-local window); (ii) the pattern-relevant component of x2 is genuinely
# global (attention-relayed code / position-0 class). This sweep separates them.
#
# Method: identical to ladder_fold.py, x2_hat from weights-only window forward (blocks 0-1)
# at W ∈ {4, 8, 16, 32, 64}; head 2.5 pattern argmax hit (positions q>=64 only, so every W
# is a strict subset of available context); null floor 0.18 (shuffled text, §1161).
#
# Registered predictions:
#   pred_a MONOTONE: hit rises with W at every step.
#   pred_b BOUNDED WINDOW SUFFICES: hit(W=64) >= 0.8 — pattern-locality is log-local, and
#          the fold program can close L2 with a wide-window known variable.
#          Alternative (flat curve, W=64 < 0.65): the gap is attention-relayed global code —
#          folding L2 requires modeling the relay, not more window.
#   pred_c LOSS/PATTERN DISSOCIATION QUANTIFIED: hit(W=8) − hit(W=4) > 0.03 even though
#          W=4 is already loss-free (+0.014) — the dissociation is graded, not a threshold.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'm1_width_results.json'
NR = 32; WS = [4, 8, 16, 32, 64]
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def window_resid(tokens, W, nblocks):
    B, Tn = tokens.shape
    idx = torch.arange(Tn, device=DEV)
    win = torch.stack([tokens[:, (idx + o).clamp_min(0)] for o in range(-(W - 1), 1)], -1)
    flat = win.reshape(B * Tn, W)
    outs = []
    step = max(256, 8192 // W)
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
    res = {f'W{w}': {'hit': 0, 'n': 0} for w in WS}
    cap = {}
    def pre(mo_, args): cap['x'] = args[0]
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous()
        h = m.transformer.h[2].attn.register_forward_pre_hook(pre)
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        h.remove()
        at = m.transformer.h[2].attn
        cos, sin = at.rotary(at.c_q(cap['x']).view(4, T, 9, 128))
        rp = head_pattern_from(cap['x'], 2, 5, cos, sin)
        for w in WS:
            xh = window_resid(idx, w, 2)
            fp = head_pattern_from(xh, 2, 5, cos, sin)
            st = res[f'W{w}']
            for b in range(4):
                for q in range(64, T, 3):
                    kr = int(rp[b, q, :q].abs().argmax()); kf = int(fp[b, q, :q].abs().argmax())
                    st['hit'] += int(kr == kf); st['n'] += 1
    H = {k: round(v['hit'] / max(v['n'], 1), 4) for k, v in res.items()}
    hits = [H[f'W{w}'] for w in WS]
    out = {'n_rows': NR, 'head': '2.5', 'hits': H, 'null_floor_ref': 0.18,
           'pred_a_monotone': bool(all(hits[j + 1] > hits[j] for j in range(len(hits) - 1))),
           'pred_b_bounded_window': bool(H['W64'] >= 0.8),
           'alt_global_relay': bool(H['W64'] < 0.65),
           'pred_c_graded_dissociation': bool(H['W8'] - H['W4'] > 0.03),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(" ".join(f"W{w}:{H[f'W{w}']}" for w in WS), flush=True)
    print(f"pred_a monotone {out['pred_a_monotone']} | pred_b bounded {out['pred_b_bounded_window']} | "
          f"alt global {out['alt_global_relay']} | pred_c graded {out['pred_c_graded_dissociation']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
