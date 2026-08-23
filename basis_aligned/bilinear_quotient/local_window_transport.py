# local_window_transport: mechanism test of §1152's scattered>contiguous inversion.
#
# Claim under test: transport is READ LOCALLY — what position t's prediction inherits from a
# patch is dominated by the patched fraction of t's trailing window (the recency kernel is
# 0.58 of the middle pool, §1099). §1152 saw the macroscopic signature (scattered half beats
# contiguous half at equal count, 0.62 vs 0.55/0.45). This experiment tests it per-position.
#
# Harness: identical to §1150-52 (residual patch after blocks L6-14, K=256, fresh rows,
# alignment-to-source readout) — but per-POSITION alignment is recorded and regressed.
#
# Conditions (all p=0.5 except full):
#   scat50    — random scattered half (per-position record + regression)
#   blocks16  — alternating 16-position blocks (patched/unpatched)
#   blocks64  — alternating 64-position blocks
#   prefix50  — contiguous first half (per-position profile: decay past the boundary)
#   full      — reference
#
# Registered predictions:
#   pred_a WINDOW REGRESSION: in scat50, per-position alignment correlates with the patched
#          fraction of the position's trailing 32-token window: Pearson r > 0.3.
#   pred_b INTERLEAVING LADDER: finer blocks transport more at fixed p=0.5:
#          align(blocks16) > align(blocks64) > align(prefix50).
#   pred_c BOUNDARY DECAY: in prefix50, unpatched-half alignment decays with distance from
#          the boundary — mean alignment of positions 128-159 exceeds that of 224-255 (the
#          transported topic leaks forward through pooling but fades).
# Controls: full reference; scat50 replicates §1152's 0.6211. Own-position patched-vs-not
# contrast reported (patched positions should align higher than unpatched in scat50).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'local_window_transport_results.json'
NEVAL = 160; SEQ = 256; REF_LAYERS = [8, 10, 12]; ABL = list(range(6, 15)); WIN = 32
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
            x = torch.where(ST['mask'], xn, x)
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
    del caps, devsum, dev, devc

    n = blocks.shape[0] // 2; src = blocks[:n].contiguous(); tgt = blocks[n:2 * n].contiguous()
    CONDS = ['scat50', 'blocks16', 'blocks64', 'prefix50', 'full']
    sum_al = {c: 0.0 for c in CONDS}; npos = 0
    # per-position accumulators
    poscos_prefix = None; nb = 0
    scat_cos_all = []; scat_frac_all = []; scat_patched_all = []
    gp = torch.Generator(device=DEV).manual_seed(1)
    for i in range(0, n, 8):
        si = src[i:i+8].to(DEV)[:, :-1].contiguous(); ti = tgt[i:i+8].to(DEV)[:, :-1].contiguous()
        if si.shape[0] != ti.shape[0]: continue
        ST['mode'] = 'cap'; ST['store'] = {}; ls = fwd(si).float(); ST['mode'] = None
        srcres = {li: ST['store'][li] for li in ABL}
        lb = fwd(ti).float()
        B, T = ti.shape
        scat = torch.zeros(B, T, dtype=torch.bool, device=DEV)
        for b in range(B):
            perm = torch.randperm(T, generator=gp, device=DEV)
            scat[b, perm[:T // 2]] = True
        pos = torch.arange(T, device=DEV)
        MASKS = {'scat50': scat,
                 'blocks16': ((pos // 16) % 2 == 0).expand(B, T),
                 'blocks64': ((pos // 64) % 2 == 0).expand(B, T),
                 'prefix50': (pos < T // 2).expand(B, T),
                 'full': torch.ones(B, T, dtype=torch.bool, device=DEV)}
        for cname in CONDS:
            ST['mode'] = 'patch'; ST['U'] = U256
            ST['srcres'] = srcres; ST['mask'] = MASKS[cname].unsqueeze(-1)
            lp = fwd(ti).float(); ST['mode'] = None
            cos = F.cosine_similarity((lp - lb), (ls - lb), dim=-1)      # (B,T)
            sum_al[cname] += float(cos.sum())
            if cname == 'prefix50':
                poscos_prefix = cos.sum(0) if poscos_prefix is None else poscos_prefix + cos.sum(0)
                nb += B
            if cname == 'scat50':
                mf = MASKS['scat50'].float()
                cw = mf.cumsum(1)
                frac = (cw - torch.cat([torch.zeros(B, WIN, device=DEV), cw[:, :-WIN]], 1)) / WIN
                keep = pos >= WIN                                         # full windows only
                scat_cos_all.append(cos[:, keep].reshape(-1).cpu())
                scat_frac_all.append(frac[:, keep].reshape(-1).cpu())
                scat_patched_all.append(MASKS['scat50'][:, keep].reshape(-1).cpu())
        npos += B * T

    al = {c: round(s / npos, 4) for c, s in sum_al.items()}
    cosv = torch.cat(scat_cos_all); fracv = torch.cat(scat_frac_all); patv = torch.cat(scat_patched_all)
    cc = torch.corrcoef(torch.stack([cosv, fracv]))[0, 1].item()
    own_p = float(cosv[patv].mean()); own_u = float(cosv[~patv].mean())
    prof = (poscos_prefix / nb)
    seg = {'128_159': round(float(prof[128:160].mean()), 4), '160_191': round(float(prof[160:192].mean()), 4),
           '192_223': round(float(prof[192:224].mean()), 4), '224_255': round(float(prof[224:].mean()), 4),
           'patched_0_127': round(float(prof[:128].mean()), 4)}
    out = {'n_positions': npos, 'alignment': al,
           'scat50_window_regression': {'pearson_r': round(cc, 4), 'window': WIN,
                                        'patched_pos_mean': round(own_p, 4), 'unpatched_pos_mean': round(own_u, 4)},
           'prefix50_profile_segments': seg,
           'pred_a_window_reads': bool(cc > 0.3),
           'pred_b_interleave_ladder': bool(al['blocks16'] > al['blocks64'] > al['prefix50']),
           'pred_c_boundary_decay': bool(seg['128_159'] > seg['224_255']),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"alignment {al}", flush=True)
    print(f"scat50 window-frac r {out['scat50_window_regression']['pearson_r']} | patched-pos {own_p:.4f} vs unpatched {own_u:.4f}")
    print(f"prefix50 profile {seg}")
    print(f"pred_a window {out['pred_a_window_reads']} | pred_b ladder {out['pred_b_interleave_ladder']} | pred_c decay {out['pred_c_boundary_decay']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
