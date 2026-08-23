# content_grain_ladder: re-ask the grain/coherence questions INSIDE the validated §1059-60 harness.
#
# §1149 nullified the entire §1146-48 sub-thread: my pipeline (deltas at 3 MLP inputs,
# donor-value broadcasts, affinity readouts) never contained a working positive control —
# rotated-basis null scored the same as the real thing. Method law: replicate the known
# positive under the modified protocol BEFORE building ladders. This script does both at once:
# it re-runs the §1059-60 protocol EXACTLY (residual patch after each block L6-14 with
# position-aligned source coords; alignment-to-source + KL readout; random-subspace null)
# on FRESH rows, and adds the grain/coherence conditions as extra patch variants at the SAME
# locus with the SAME readout — so every rung is judged by an instrument that provably
# sees the known positive.
#
# Conditions (all: replace target coords in span U with source-derived coords, after each
# block L6-14, forward mechanics copied verbatim from content_patching.py):
#   c8 / c16 / c64 / c256  — content-basis rank ladder (16/64/256 replicate §1060's sweep:
#                            alignment 0.5783 / 0.7696 / 0.8995; rank 8 is new)
#   r256                   — random 256-dim subspace null (§1059: 0.6689)
#   shuf256                — K=256 content coords, source positions PERMUTED along time
#                            (ONE permutation per batch, shared across layers, so the patch
#                            is a coherently-WRONG arrangement, not inter-layer noise)
#   mean256                — per-sequence MEAN source coord broadcast to all positions
#                            (the 'uniform broadcast' question, now in-protocol)
#
# Registered predictions:
#   pred_0 REPLICATES: align(c256) > 0.8 and exceeds align(r256) by > 0.15; align(c16) > 0.45.
#           If this fails the thread stops — the positive itself didn't reproduce on fresh
#           rows and nothing downstream is interpretable.
#   pred_a  grain is coarse: align(c8) >= 0.7 * align(c16).
#   pred_b  coherence matters: align(shuf256) < align(c256) - 0.15.
#   pred_c  one average vector is NOT enough: align(mean256) <= align(shuf256) + 0.05.
# Controls/nulls: r256 is the null for every content rung; shuf/mean are graded internal
# controls of c256. Fresh rows = fineweb_rows(2N)[N:] so this is a replication, not a rerun.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_grain_ladder_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15))
ST = {'mode': None, 'U': None, 'srcres': None, 'store': None, 'variant': None, 'perm': None}


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for li, blk in enumerate(m.transformer.h):
        x, v1 = blk(x, v1, x0)
        if ST['mode'] == 'cap' and li in ABL:
            ST['store'][li] = x.detach()
        elif ST['mode'] == 'patch' and li in ABL:
            U = ST['U']; xs = ST['srcres'][li]
            c = xs @ U                                     # source coords (B,T,K)
            if ST['variant'] == 'shuf':
                c = c[:, ST['perm'], :]
            elif ST['variant'] == 'mean':
                c = c.mean(1, keepdim=True).expand_as(c)
            x = x - (x @ U) @ U.T + c @ U.T
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
    rows = cl.fineweb_rows(2 * NEVAL)[NEVAL:]              # FRESH half -> replication, not rerun
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])

    # content basis from pooled L8-12 per-token deviation, built on the fresh rows (in-protocol)
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

    CONDS = [('c8', U256[:, :8].contiguous(), None), ('c16', U256[:, :16].contiguous(), None),
             ('c64', U256[:, :64].contiguous(), None), ('c256', U256, None),
             ('r256', R256, None), ('shuf256', U256, 'shuf'), ('mean256', U256, 'mean')]

    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    acc = {name: {'kl': 0.0, 'al': 0.0} for name, _, _ in CONDS}; npos = 0
    gp = torch.Generator(device=DEV).manual_seed(1)
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        lb = fwd(ti).float()
        base = F.log_softmax(lb, -1)
        perm = torch.randperm(si.shape[1], generator=gp, device=DEV)   # one perm/batch, all layers
        for name, U, variant in CONDS:
            ST['mode'] = 'patch'; ST['U'] = U; ST['srcres'] = srcres
            ST['variant'] = variant; ST['perm'] = perm
            lp = fwd(ti).float(); ST['mode'] = None
            patch = F.log_softmax(lp, -1)
            kl = (patch.exp() * (patch - base)).sum(-1)
            cos = F.cosine_similarity((lp - lb).reshape(-1, V), (ls - lb).reshape(-1, V), dim=-1)
            acc[name]['kl'] += float(kl.sum()); acc[name]['al'] += float(cos.sum())
        npos += si.shape[0] * si.shape[1]

    res = {name: {'kl': round(a['kl']/npos, 4), 'alignment': round(a['al']/npos, 4)}
           for name, a in acc.items()}
    al = {k: v['alignment'] for k, v in res.items()}
    out = {'n_positions': npos, 'abl_range': [ABL[0], ABL[-1]], 'conds': res,
           'orig_1059_60': {'c256': 0.8995, 'r256': 0.6689, 'c16': 0.5783, 'c64': 0.7696},
           'pred_0_replicates': bool(al['c256'] > 0.8 and al['c256'] - al['r256'] > 0.15 and al['c16'] > 0.45),
           'pred_a_grain_coarse': bool(al['c8'] >= 0.7 * al['c16']),
           'pred_b_coherence_matters': bool(al['shuf256'] < al['c256'] - 0.15),
           'pred_c_mean_not_enough': bool(al['mean256'] <= al['shuf256'] + 0.05),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for name, _, _ in CONDS:
        print(f"{name:>8}: KL {res[name]['kl']:7.3f} | align->source {res[name]['alignment']:+.4f}", flush=True)
    print(f"pred_0 replicates {out['pred_0_replicates']} | pred_a grain {out['pred_a_grain_coarse']} | "
          f"pred_b coherence {out['pred_b_coherence_matters']} | pred_c mean {out['pred_c_mean_not_enough']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
