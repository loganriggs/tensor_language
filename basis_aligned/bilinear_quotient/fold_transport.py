# fold_transport: do the session's two closed arcs COMPOSE? Content transport (§1150-60)
# re-run inside the pattern-folded model (§1166).
#
# If topic transport (residual patching of content coords, L6-14) survives unchanged when
# every attention pattern is a window-folded reconstruction, then transport genuinely does
# not depend on exact selection details — the two laws are independent layers of the same
# account (selection = bounded-window function; content = per-position transported state).
# If transport degrades, the fine pattern residue (§1145's interaction-borne remainder)
# is load-bearing for addressing after all — either answer is informative.
#
# Harness: fold_pattern_loss2's custom forward (all 162 patterns folded, W=128, exact-prefix)
# + §1150's patching (replace target content coords with position-aligned source coords after
# each block L6-14) + §1150's readout (alignment of logit deltas to source, KL), fresh rows.
# Both source-capture and target runs use the SAME folded forward (consistent counterfactual).
#
# Conditions: c256 (content basis), r256 (random-subspace null) — §1150 references 0.8994 /
# 0.6693 under the live forward.
#
# Registered predictions:
#   pred_a TRANSPORT SURVIVES FOLDING: |align(c256)_folded − 0.8994| <= 0.05.
#   pred_b SPECIFICITY SURVIVES: align(c256) − align(r256) >= 0.18 (live: 0.23).
#   pred_c SANITY: the folded model's base CE on these rows ≈ 3.377 (§1166's fold_all CE),
#          within 0.05 (fresh-draw tolerance).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fold_transport_results.json'
NEVAL = 160; SEQ = 256; W = 128
REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None}


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
    res = torch.cat(outs, 0).reshape(B, Tn, D)
    Wp = min(W, Tn)
    xp = F.rms_norm(m.transformer.wte(tokens[:, :Wp]), (D,)); x0p = xp; v1p = None
    for blk in m.transformer.h[:nblocks]:
        xp, v1p = blk(xp, v1p, x0p)
    res[:, :Wp] = xp.detach()
    return res


def pattern_from(xin, at, cos, sin):
    B = xin.shape[0]
    q = F.rms_norm(at.c_q(xin).view(B, T, 9, 128), (128,))
    k = F.rms_norm(at.c_k(xin).view(B, T, 9, 128), (128,))
    q2 = F.rms_norm(at.c_q2(xin).view(B, T, 9, 128), (128,))
    k2 = F.rms_norm(at.c_k2(xin).view(B, T, 9, 128), (128,))
    q = are(q, cos, sin); k = are(k, cos, sin); q2 = are(q2, cos, sin); k2 = are(k2, cos, sin)
    pat = (torch.einsum('bqhd,bkhd->bhqk', q.float(), k.float()) / 128.0) \
        * (torch.einsum('bqhd,bkhd->bhqk', q2.float(), k2.float()) / 128.0)
    mask = torch.tril(torch.ones(T, T, device=DEV, dtype=torch.bool))
    return pat.masked_fill(~mask, 0.0)


@torch.no_grad()
def folded_fwd(idx, XH):
    """All-pattern-folded forward with §1150 cap/patch hooks on the residual after ABL blocks."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    B = idx.shape[0]
    for L, blk in enumerate(m.transformer.h):
        at = blk.attn
        xm = blk.lambdas[0] * x + blk.lambdas[1] * x0
        xin = F.rms_norm(xm, (D,))
        cos, sin = at.rotary(at.c_q(xin).view(B, T, 9, 128))
        pat = pattern_from(F.rms_norm(blk.lambdas[0] * XH[L] + blk.lambdas[1] * x0, (D,)), at, cos, sin)
        v = at.c_v(xin).view(B, T, 9, 128)
        if v1 is None:
            v1 = v
        vv = (1 - at.lamb) * v + at.lamb * v1.view_as(v)
        y = torch.einsum('bhqk,bkhd->bqhd', pat.to(vv.dtype), vv).reshape(B, T, D)
        x = xm + at.c_proj(y)
        x = x + blk.mlp(F.rms_norm(x, (D,)))
        if ST['mode'] == 'cap' and L in ABL:
            ST['store'][L] = x.detach()
        elif ST['mode'] == 'patch' and L in ABL:
            U = ST['U']; xs = ST['srcres'][L]
            x = x - (x @ U) @ U.T + (xs @ U) @ U.T
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def capture_dev(blocks):
    caps = {L: [] for L in REF_LAYERS}; toks = []; hs = []
    for L in REF_LAYERS:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_):
                caps[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    # NOTE: hooks fire on blk.mlp inside folded_fwd too (we call blk.mlp directly)
    ST['mode'] = None
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i + 8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1))
        XH = {L: window_resid(idx, W, L) for L in range(18)}   # per-chunk (memory-safe)
        folded_fwd(idx, XH)
        del XH
    for h in hs: h.remove()
    return {L: torch.cat(caps[L], 0) for L in REF_LAYERS}, torch.cat(toks, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(2 * NEVAL)[NEVAL:]
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])

    caps, tok = capture_dev(blocks)   # window residuals computed per-chunk inside (memory-safe)
    devsum = None
    for L in REF_LAYERS:
        X = caps[L]; xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dv = X - xbar[tok]; devsum = dv if devsum is None else devsum + dv; del X, dv
    dev = devsum / len(REF_LAYERS); devc = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(devc, full_matrices=False)
    U256 = Vt[:256].T.contiguous()
    g = torch.Generator(device=DEV).manual_seed(0)
    R256 = torch.linalg.qr(torch.randn(D, 256, generator=g, device=DEV))[0]
    del caps, devsum, dev, devc

    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    acc = {'c256': {'kl': 0.0, 'al': 0.0}, 'r256': {'kl': 0.0, 'al': 0.0}}
    base_ce = 0.0; npos = 0; ntok_ce = 0
    for i in range(0, n, 8):
        si = src[i:i + 8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i + 8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        XHsrc = {L: window_resid(si, W, L) for L in range(18)}
        XHtgt = {L: window_resid(ti, W, L) for L in range(18)}
        ST['mode'] = 'cap'; ST['store'] = {}; ls = folded_fwd(si, XHsrc).float(); ST['mode'] = None
        srcres = {L: ST['store'][L] for L in ABL}
        lb = folded_fwd(ti, XHtgt).float()
        tt = tgt[i:i + 8].to(DEV)[:, 1:].contiguous()
        base_ce += float(F.cross_entropy(lb.reshape(-1, V), tt.reshape(-1), reduction='sum')); ntok_ce += tt.numel()
        base = F.log_softmax(lb, -1)
        for name, U in (('c256', U256), ('r256', R256)):
            ST['mode'] = 'patch'; ST['U'] = U; ST['srcres'] = srcres
            lp = folded_fwd(ti, XHtgt).float(); ST['mode'] = None
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb).reshape(-1, V), (ls - lb).reshape(-1, V), dim=-1)
            acc[name]['kl'] += float(kl.sum()); acc[name]['al'] += float(cos.sum())
        del XHsrc, XHtgt
        npos += si.shape[0] * si.shape[1]

    res = {name: {'kl': round(a['kl'] / npos, 4), 'alignment': round(a['al'] / npos, 4)}
           for name, a in acc.items()}
    base_ce = round(base_ce / ntok_ce, 4)
    al_c = res['c256']['alignment']; al_r = res['r256']['alignment']
    out = {'n_positions': npos, 'folded_base_ce': base_ce, 'conds': res,
           'live_refs_1150': {'c256': 0.8994, 'r256': 0.6693},
           'pred_a_transport_survives': bool(abs(al_c - 0.8994) <= 0.05),
           'pred_b_specificity_survives': bool(al_c - al_r >= 0.18),
           'pred_c_sanity_ce': bool(abs(base_ce - 3.377) <= 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"folded base CE {base_ce} (ref 3.377)")
    print(f"c256: KL {res['c256']['kl']} | align {al_c} (live 0.8994)")
    print(f"r256: KL {res['r256']['kl']} | align {al_r} (live 0.6693)")
    print(f"pred_a survives {out['pred_a_transport_survives']} | pred_b specificity {out['pred_b_specificity_survives']} | pred_c sanity {out['pred_c_sanity_ce']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
