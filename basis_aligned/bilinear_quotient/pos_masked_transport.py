# pos_masked_transport: WHERE does the position-bound address live?
#
# §1150 established (in-protocol) that content transport is position-bound: shuffling which
# position holds which content vector destroys all excess over the random-subspace null.
# Next rung: are all positions equal carriers, or do CONTENT-WORD positions carry the address
# disproportionately? (Links to the generation phenotype: content-band ablation collapses
# content-word rate 1.6-6.9x across the family — if the address rides content-word positions,
# the two results share one mechanism.)
#
# Harness: identical to content_grain_ladder.py / content_patching.py (residual patch after
# each block L6-14, position-aligned source coords at K=256, alignment-to-source + KL readout,
# fresh rows) — the validated instrument. New knob: a per-position MASK selects WHICH target
# positions get patched; unpatched positions keep their own coords.
#
# Position classes: frequency split on the eval corpus — the 128 most frequent token TYPES
# (function words, punctuation, whitespace-words) = 'func'; everything else = 'content'.
# Heuristic, but matches the content-word-rate instrument used in the generation phenotypes.
#
# Conditions (all K=256 content basis unless noted):
#   full    — every position patched (§1150 reference, expect align ≈ 0.90)
#   cpos    — only content-word positions patched
#   fpos    — only function-word positions patched
#   rand_c  — random position subset, per-sequence count-matched to cpos
#   rand_f  — random position subset, per-sequence count-matched to fpos
#   r256    — full-position random-subspace null (expect ≈ 0.67)
#
# Registered predictions (excess judged against the count-matched random-position controls):
#   pred_a  partial patching transports: align(cpos) and align(fpos) both > align(r256) is NOT
#           required — but at least one count-matched advantage (cond − rand_matched) > 0.03.
#   pred_b  content positions are the privileged carriers: (cpos − rand_c) > 2*(fpos − rand_f).
#   pred_c  no free lunch: full > max(cpos, fpos) — the address is distributed, partial
#           patching cannot reach full transport.
# Alternative outcomes: if (fpos − rand_f) >= (cpos − rand_c), the address rides the frequent
# scaffold positions — would connect to skeleton/grammar coupling instead; report plainly.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'pos_masked_transport_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15)); NFUNC = 128
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'mask': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ABL:
            U = ST['U']; xs = ST['srcres'][li]
            xn = x - (x @ U) @ U.T + (xs @ U) @ U.T
            M = ST['mask']                                  # (B,T,1) bool: True = patch here
            x = torch.where(M, xn, x)
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
    rows = cl.fineweb_rows(2 * NEVAL)[NEVAL:]              # same fresh half as §1150
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])

    # frequency split: top-NFUNC token types in the eval corpus = 'func'
    flat = blocks[:, :-1].reshape(-1)
    cnts_v = torch.bincount(flat, minlength=V)
    func_ids = torch.topk(cnts_v, NFUNC).indices
    is_func = torch.zeros(V, dtype=torch.bool); is_func[func_ids.cpu()] = True
    frac_func = float(is_func[flat].float().mean())

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

    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    CONDS = ['full', 'cpos', 'fpos', 'rand_c', 'rand_f', 'r256']
    acc = {c: {'kl': 0.0, 'al': 0.0} for c in CONDS}; npos = 0
    gp = torch.Generator(device=DEV).manual_seed(1)
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        lb = fwd(ti).float(); base = F.log_softmax(lb, -1)
        B, T = ti.shape
        mf = is_func.to(DEV)[ti]                            # (B,T) func-word positions of TARGET
        mc = ~mf
        # per-row count-matched random masks
        rc = torch.zeros_like(mc); rf = torch.zeros_like(mf)
        for b in range(B):
            perm = torch.randperm(T, generator=gp, device=DEV)
            kc = int(mc[b].sum()); kf = int(mf[b].sum())
            rc[b, perm[:kc]] = True; rf[b, perm[:kf]] = True
        MASKS = {'full': torch.ones_like(mc), 'cpos': mc, 'fpos': mf,
                 'rand_c': rc, 'rand_f': rf, 'r256': torch.ones_like(mc)}
        for cname in CONDS:
            ST['mode'] = 'patch'; ST['U'] = R256 if cname == 'r256' else U256
            ST['srcres'] = srcres; ST['mask'] = MASKS[cname].unsqueeze(-1)
            lp = fwd(ti).float(); ST['mode'] = None
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb).reshape(-1, V), (ls - lb).reshape(-1, V), dim=-1)
            acc[cname]['kl'] += float(kl.sum()); acc[cname]['al'] += float(cos.sum())
        npos += B * T

    res = {c: {'kl': round(a['kl']/npos, 4), 'alignment': round(a['al']/npos, 4)}
           for c, a in acc.items()}
    al = {k: v['alignment'] for k, v in res.items()}
    adv_c = round(al['cpos'] - al['rand_c'], 4); adv_f = round(al['fpos'] - al['rand_f'], 4)
    out = {'n_positions': npos, 'frac_func_positions': round(frac_func, 4), 'n_func_types': NFUNC,
           'conds': res, 'adv_content_pos': adv_c, 'adv_func_pos': adv_f,
           'pred_a_partial_transports': bool(max(adv_c, adv_f) > 0.03),
           'pred_b_content_pos_privileged': bool(adv_c > 2 * adv_f),
           'pred_c_distributed': bool(al['full'] > max(al['cpos'], al['fpos'])),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for c in CONDS:
        print(f"{c:>7}: KL {res[c]['kl']:7.3f} | align->source {res[c]['alignment']:+.4f}", flush=True)
    print(f"func-pos fraction {out['frac_func_positions']} | adv content {adv_c} | adv func {adv_f}")
    print(f"pred_a partial {out['pred_a_partial_transports']} | pred_b content-priv {out['pred_b_content_pos_privileged']} | "
          f"pred_c distributed {out['pred_c_distributed']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
