# ladder_fold: interpret LAYER-2 attention by folding weights over the KNOWN front variables.
#
# User question this answers: "why can't we fully interpret L0/L1 heads by folding in the
# weights, and L2 by folding with the known variables from previous layers?" State of the
# ledger: L0 heads ARE exactly folded (writeup 477: weights-only replacement −0.000 nats);
# the front is a BIGRAM function (480: fold exact, window-1 +0.004); block 1 is 4-local
# (481: +0.014). Prior fold attempts at L2/L3 heads fed only the UNIGRAM fold code
# (m0(rms_norm(wte))) and hit 12-39% (fold_score_test2/3); fold_gap_locate2 showed real
# residuals close the gap (D-arm hit 1.0) and named the mediator m1 (writeup 396: 2.5's
# trigger is m1-mediated; 397-399: one identity code, MLP-built, attention-moved).
#
# The missing step, done here: build x2_hat by a WEIGHTS-ONLY WINDOW FORWARD — run blocks
# 0-1 on each position's last-W tokens only (W=4; the 481-certified n-gram approximation;
# rotary is relative so window positions are exact) — and score head 2.5's attention pattern
# from folded QK forms on x2_hat. Same for 3.5 from x3_hat (blocks 0-2, W=8; block-2
# locality not separately certified — exploratory). If the hit rate approaches the real-x
# ceiling, layer-2 attention is FULLY interpreted as: weights applied to a known, enumerable
# 4-gram variable. No live-context forward enters the prediction side.
#
# Conditions per head: fold from x_hat(W) for W in {2, 4[, 8]}; null = x_hat from shuffled
# rows (wrong text, right machinery). Baselines on record: unigram fold hit 0.12-0.39;
# real-x ceiling 1.0.
#
# Registered predictions:
#   pred_a 2.5 CLOSES: hit(W=4) >= 0.85 (the 4-gram variable is enough — layer-2 trigger
#          fully weights+known-variable interpretable).
#   pred_b CONTEXT BEYOND BIGRAM NEEDED: hit(W=4) − hit(W=2) >= 0.15 on 2.5 (m1's context
#          window is the mediator, consistent with 478's correction).
#   pred_c 3.5 SUBSTANTIALLY CLOSES: hit(W=8 via blocks 0-2) >= 0.7 (exploratory bar).
# Null: shuffled-row x_hat hit <= 0.1 on both heads.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ladder_fold_results.json'
NR = 32
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb


@torch.no_grad()
def window_resid(tokens, W, nblocks):
    """Weights-only n-gram forward: residual after blocks[0:nblocks] for each position,
    computed from that position's last-W tokens only. tokens: (B,T). Returns (B,T,D)."""
    B, Tn = tokens.shape
    idx = torch.arange(Tn, device=DEV)
    win = torch.stack([tokens[:, (idx + o).clamp_min(0)] for o in range(-(W - 1), 1)], -1)  # (B,T,W)
    # left-pad short prefixes by clamping (positions < W-1 repeat token 0 — matches their
    # real prefix poorly only for q < W, which the scorer skips anyway (q >= 8))
    flat = win.reshape(B * Tn, W)
    outs = []
    for i in range(0, flat.shape[0], 2048):
        wb = flat[i:i + 2048]
        x = F.rms_norm(m.transformer.wte(wb), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h[:nblocks]:
            x, v1 = blk(x, v1, x0)
        outs.append(x[:, -1].detach())
    return torch.cat(outs, 0).reshape(B, Tn, D)


@torch.no_grad()
def head_pattern_from(X, li, hd, cos, sin):
    """Squared-attention pattern of head (li,hd) computed from residuals X (B,T,D)."""
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
    HEADS = [(2, 5), (3, 5)]
    # conditions: (tag, head, W, nblocks, shuffle)
    ARMS = [('h25_W4', (2, 5), 4, 2, False), ('h25_W2', (2, 5), 2, 2, False),
            ('h25_null', (2, 5), 4, 2, True),
            ('h35_W8', (3, 5), 8, 3, False), ('h35_W2', (3, 5), 2, 3, False),
            ('h35_null', (3, 5), 8, 3, True)]
    res = {tag: {'hit': 0, 'n': 0} for tag, _, _, _, _ in ARMS}
    cap = {}
    def mkpre(li):
        def h(mo_, args): cap[li] = args[0]
        return h
    for i in range(0, NR, 4):
        bb = ROWS[i:i + 4].to(DEV); idx = bb[:, :-1].contiguous()
        hs = [m.transformer.h[li].attn.register_forward_pre_hook(mkpre(li)) for li, _ in HEADS]
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        for h in hs: h.remove()
        at0 = m.transformer.h[2].attn
        cos, sin = at0.rotary(at0.c_q(cap[2]).view(4, T, 9, 128))
        real = {(li, hd): head_pattern_from(cap[li], li, hd, cos, sin) for li, hd in HEADS}
        xhat_cache = {}
        for tag, (li, hd), W, nb, shuf in ARMS:
            key = (W, nb, shuf)
            if key not in xhat_cache:
                toks = idx if not shuf else idx[torch.randperm(4, device=DEV)]
                xhat_cache[key] = window_resid(toks, W, nb)
            fpat = head_pattern_from(xhat_cache[key], li, hd, cos, sin)
            rp = real[(li, hd)]
            for b in range(4):
                for q in range(8, T, 3):
                    kr = int(rp[b, q, :q].abs().argmax()); kf = int(fpat[b, q, :q].abs().argmax())
                    res[tag]['hit'] += int(kr == kf); res[tag]['n'] += 1
    H = {tag: round(v['hit'] / max(v['n'], 1), 4) for tag, v in res.items()}
    out = {'n_rows': NR, 'hits': H,
           'baselines': {'unigram_fold_prior': '0.12-0.39 (fold_score_test2/3)', 'real_x_ceiling': 1.0},
           'pred_a_25_closes': bool(H['h25_W4'] >= 0.85),
           'pred_b_beyond_bigram': bool(H['h25_W4'] - H['h25_W2'] >= 0.15),
           'pred_c_35_closes': bool(H['h35_W8'] >= 0.7),
           'null_ok': bool(H['h25_null'] <= 0.1 and H['h35_null'] <= 0.1),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for tag in H:
        print(f"{tag:>9}: hit {H[tag]}", flush=True)
    print(f"pred_a 2.5 closes {out['pred_a_25_closes']} | pred_b beyond-bigram {out['pred_b_beyond_bigram']} | "
          f"pred_c 3.5 {out['pred_c_35_closes']} | null ok {out['null_ok']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
