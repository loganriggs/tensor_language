"""FIX of §1097's invalid retention metric (registered there). Same question — which attention ensemble's
dynamics preserve the CONTENT SEED — but measured magnitude-invariantly: per-position ALIGNMENT between the
condition run's content coordinates and the full run's. c_full = U_c^T (x_L8 - xbar[tok]) from the FULL model;
c_cond = same from the condition run (same tokens, same xbar, same U_c). Alignment = mean per-position cosine
(c_cond, c_full) + R^2 = 1 - ||c_cond - c_full||^2 / ||c_full||^2 (reported both; cosine is the headline —
strictly magnitude-invariant). Conditions as §1097: all-const except {front L0-2 | gatherer L3-5 |
gatherer-minus-5.7 | middle L6-9 | random-27 | none}. NSEQ=192.

REGISTERED PREDICTIONS:
  (0) SANITY: base (keep-all) cosine ~1; none_allconst the lowest cosine; no value > 1 possible (metric fixed).
  (a) SEED IN THE GATHERER: keeping L3-5 dynamics gives the highest content alignment of any single band
      (cosine gain over none >= 2x front's gain), and dropping 5.7 changes it little -> the seed is gathered
      by the L3-5 pooler ensemble;
  (b) BREADTH NEEDED (per §1097 CE): if no band beats none by much and only wide ensembles align, the seed
      itself is super-additively gathered (report plainly);
  (c) CE and alignment RANK-AGREE across conditions better than §1097's broken metric did."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_seed_align_results.json'
NSEQ = 192; SEQ = 256; REF = [8, 10, 12]; K = 64; L_PROBE = 8
H = m.transformer.h
CTL = {'keep': None}
MEANS = {}
CAP8 = []


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def hook(L):
    def h(mo, args):
        if CTL['keep'] is None: return None
        y = args[0].clone()
        for hh in range(NH):
            if (L, hh) not in CTL['keep']:
                y[..., hh*HD:(hh+1)*HD] = MEANS[L][hh].view(1, 1, HD).to(y.dtype)
        return (y,) + tuple(args[1:])
    return h


def cap8_hook(mo, i_, o_):
    CAP8.append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
    return None


@torch.no_grad()
def run_condition(blocks, keep, xbar8, Uc, c_full):
    CTL['keep'] = keep; CAP8.clear()
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    CTL['keep'] = None
    X = torch.cat(CAP8, 0); CAP8.clear()
    tok = ST_TOK
    c_cond = (X - xbar8[tok]) @ Uc
    cos = float(F.cosine_similarity(c_cond, c_full, dim=-1).mean())
    r2 = 1 - float(((c_cond - c_full)**2).sum()) / float((c_full**2).sum())
    return tot/n, cos, r2


@torch.no_grad()
def main():
    global ST_TOK
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])

    # pass 1 (full model): head means, content basis, xbar8, full-run content coords
    caps = {L: torch.zeros(NH, HD, device=DEV) for L in range(18)}
    hs = []
    for L in range(18):
        def mk(L):
            def h(mo, args): caps[L] += args[0].detach().float().reshape(-1, NH, HD).sum(0)
            return h
        hs.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    capR = {Lr: [] for Lr in REF}
    for Lr in REF:
        def mkr(Lr):
            def h(mo, i_, o_): capR[Lr].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(H[Lr].mlp.register_forward_hook(mkr(Lr)))
    idsL = []; npos = 0
    for i in range(0, NSEQ, 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); idsL.append(idx.reshape(-1)); fwd(idx); npos += idx.numel()
    for h in hs: h.remove()
    for L in range(18): MEANS[L] = caps[L] / npos
    ST_TOK = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, ST_TOK, torch.ones_like(ST_TOK, dtype=torch.float))
    devsum = None; xbar8 = None; X8_full = None
    for Lr in REF:
        X = torch.cat(capR[Lr], 0); capR[Lr] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, ST_TOK, X)
        xb = xb/cn.clamp_min(1).unsqueeze(1)
        if Lr == L_PROBE: xbar8 = xb.clone(); X8_full = X.clone()
        dv = X - xb[ST_TOK]
        devsum = dv if devsum is None else devsum + dv
        del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False); Uc = Vt[:K].T.contiguous()
    c_full = (X8_full - xbar8[ST_TOK]) @ Uc
    del dev, devsum, X8_full

    hs = [H[L].attn.c_proj.register_forward_pre_hook(hook(L)) for L in range(18)]
    hc = H[L_PROBE].mlp.register_forward_hook(cap8_hook)
    allheads = {(L, h2) for L in range(18) for h2 in range(NH)}
    base_ce, base_cos, base_r2 = run_condition(blocks, allheads, xbar8, Uc, c_full)
    g = torch.Generator().manual_seed(0)
    rand27 = set()
    while len(rand27) < 27:
        rand27.add((int(torch.randint(3, 15, (1,), generator=g)), int(torch.randint(0, NH, (1,), generator=g))))
    gather = {(L, h2) for L in (3, 4, 5) for h2 in range(NH)}
    conds = {'front_L0_2': {(L, h2) for L in (0, 1, 2) for h2 in range(NH)},
             'gatherer_L3_5': gather, 'gatherer_minus_57': gather - {(5, 7)},
             'middle_L6_9': {(L, h2) for L in (6, 7, 8, 9) for h2 in range(NH)},
             'random_27': rand27, 'none_allconst': set()}
    out = {'base': {'ce': round(base_ce, 4), 'cos': round(base_cos, 4), 'r2': round(base_r2, 4)}, 'conditions': {}}
    for name, keep in conds.items():
        c, cos, r2 = run_condition(blocks, keep, xbar8, Uc, c_full)
        out['conditions'][name] = {'ce_cost': round(c-base_ce, 4), 'cos': round(cos, 4), 'r2': round(r2, 4)}
        print(f"{name:>20}: CE +{c-base_ce:.4f} | content cos {cos:.4f} | R2 {r2:.4f}", flush=True)
    for h in hs: h.remove()
    hc.remove()

    cd = out['conditions']
    nc = cd['none_allconst']['cos']
    def gain(n): return cd[n]['cos'] - nc
    out['pred_a_seed_in_gatherer'] = bool(gain('gatherer_L3_5') >= 2*gain('front_L0_2')
                                          and gain('gatherer_L3_5') > 0.05
                                          and abs(cd['gatherer_minus_57']['cos'] - cd['gatherer_L3_5']['cos']) < 0.05)
    gains = {n: gain(n) for n in conds if n != 'none_allconst'}
    out['pred_b_breadth_needed'] = bool(max(gains.values()) < 0.1)
    ranks_ce = sorted(conds, key=lambda n: cd[n]['ce_cost'])
    ranks_cos = sorted(conds, key=lambda n: -cd[n]['cos'])
    out['rank_agreement'] = ranks_ce == ranks_cos
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"gains over all-const: { {k: round(v,4) for k,v in gains.items()} }", flush=True)
    print(f"pred_a gatherer-seed {out['pred_a_seed_in_gatherer']} | pred_b breadth {out['pred_b_breadth_needed']} | rank-agree {out['rank_agreement']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
