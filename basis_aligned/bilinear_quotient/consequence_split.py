# consequence_split: does half of content transport ride PROCESSED CONSEQUENCES rather than
# the final content coordinates? (§1155's registered decisive test.)
#
# §1155 found: a single L14 patch delivers 0.96 of the full clamp's FINAL content coords but
# only 50% of its logit excess. Hypothesis: dense patching lets every intermediate block
# compute on source content, writing the source's in-flight products into components OUTSIDE
# the K=256 content span, which the readout also consumes.
#
# Test: patch as usual (full9 = L6-14 / last1 = L14 only), then at the FINAL residual
# (after block 17, before the readout) STRIP the content coords back to the target-base
# run's own values. What remains of transport can only be carried outside the content span.
#
# Conditions (K=256, full position coverage, fresh rows, §1150-55 harness):
#   full9        — reference (0.8994)
#   last1        — reference (0.7843)
#   full9_strip  — patch L6-14, final content coords reset to base
#   last1_strip  — patch L14 only, final content coords reset to base
#   strip_only   — no patch, final coords reset to base (null: strip operation itself ≈ 0)
#   r256         — full-coverage random-subspace null (0.6693)
#
# Registered predictions (excess = alignment − r256):
#   pred_a CONSEQUENCES REAL: excess(full9_strip) >= 0.35 × excess(full9).
#   pred_b LAST1 IS COORDS-ONLY: excess(last1_strip) <= 0.03.
#   pred_c ADDITIVE: |excess(full9) − (excess(full9_strip) + excess(last1))| <= 0.05.
# Null: strip_only should sit near 0 excess (the strip itself is nearly free on the base run
# because it replaces base coords with themselves — it differs only via fp noise; report it).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'consequence_split_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'layers': None,
      'strip': False, 'base_final': None, 'grab_final': False, 'final': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ST['layers']:
            U = ST['U']; xs = ST['srcres'][li]
            x = x - (x @ U) @ U.T + (xs @ U) @ U.T
    if ST['grab_final']:
        ST['final'] = x.detach()
    if ST['strip']:
        U = ST['U']; xb = ST['base_final']
        x = x - (x @ U) @ U.T + (xb @ U) @ U.T
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


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
    ST['mode'] = None
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1)); fwd(idx)
    for h in hs: h.remove()
    return {L: torch.cat(caps[L], 0) for L in REF_LAYERS}, torch.cat(toks, 0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(2 * NEVAL)[NEVAL:]
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])

    caps, tok = capture_dev(blocks)
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

    L69, L14 = set(range(6, 15)), {14}
    CONDS = [('full9', L69, U256, False), ('last1', L14, U256, False),
             ('full9_strip', L69, U256, True), ('last1_strip', L14, U256, True),
             ('strip_only', set(), U256, True), ('r256', L69, R256, False)]
    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    acc = {c: {'kl': 0.0, 'al': 0.0} for c, _, _, _ in CONDS}; npos = 0
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        # base run: also grab its final residual for the strip target
        ST['grab_final'] = True; lb = fwd(ti).float(); ST['grab_final'] = False
        base_final = ST['final']; base = F.log_softmax(lb, -1)
        for cname, layers, U, strip in CONDS:
            ST['mode'] = 'patch'; ST['U'] = U; ST['srcres'] = srcres; ST['layers'] = layers
            ST['strip'] = strip; ST['base_final'] = base_final
            lp = fwd(ti).float(); ST['mode'] = None; ST['strip'] = False
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb).reshape(-1, V), (ls - lb).reshape(-1, V), dim=-1)
            acc[cname]['kl'] += float(kl.sum()); acc[cname]['al'] += float(cos.sum())
        npos += si.shape[0] * si.shape[1]

    res = {c: {'kl': round(a['kl']/npos, 4), 'alignment': round(a['al']/npos, 4)}
           for c, a in acc.items()}
    al = {k: v['alignment'] for k, v in res.items()}; r = al['r256']
    exc = {k: round(v - r, 4) for k, v in al.items()}
    out = {'n_positions': npos, 'conds': res, 'excess_over_r256': exc,
           'pred_a_consequences_real': bool(exc['full9_strip'] >= 0.35 * exc['full9']),
           'pred_b_last1_coords_only': bool(exc['last1_strip'] <= 0.03),
           'pred_c_additive': bool(abs(exc['full9'] - (exc['full9_strip'] + exc['last1'])) <= 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for c, _, _, _ in CONDS:
        print(f"{c:>12}: KL {res[c]['kl']:7.3f} | align {res[c]['alignment']:+.4f} | excess {exc[c]:+.4f}", flush=True)
    print(f"pred_a consequences {out['pred_a_consequences_real']} | pred_b coords-only {out['pred_b_last1_coords_only']} | pred_c additive {out['pred_c_additive']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
