"""FAN-OUT A (dossier attn-middle-pooling, Open A): WHICH attention ensemble authors the content seed?
§1074: early attention creates content-R2 gains; §1089 retired L5H7 (constant); §1093: dynamics matter
collectively. Ensemble-level test: run with ALL 162 heads static (global-mean const) EXCEPT one kept-dynamic band;
measure (1) CE and (2) CONTENT RETENTION — the fraction of the full model's deep content signal that survives:
capture L8 MLP input in the condition run, compute its per-token deviation, project onto the FULL run's content
basis U_c (top-64 of pooled L8-12 deviation), retained = ||proj||^2 / full-run baseline. Bands kept dynamic:
front L0-2 | gatherer L3-5 | gatherer-minus-5.7 | middle L6-9 | random-27-heads (matched count) | none (all-const)
| all (base). NSEQ=192 (data-scale per user directive).

REGISTERED PREDICTIONS:
  (0) SANITY: base retention ~1; all-const retention lowest; random-27 between none and the real bands.
  (a) THE SEED IS GATHERED IN L3-5: keeping gatherer dynamics retains >= 0.6 of base content (vs all-const)
      while front-only retains < half of what gatherer does; excluding 5.7 from the gatherer changes ~nothing
      (it is a constant, §1089) -> the content seed's carriers are the L3-5 pooler ensemble minus the sink;
  (b) if middle L6-9 retains as much as L3-5, the seed has no privileged band (gathering is depth-distributed;
      report plainly);
  (c) CE and content retention should RANK TOGETHER across conditions (the collective §1093 cost is largely
      the content function)."""
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; NH = 9; HD = 128; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_seed_ensembles_results.json'
NSEQ = 192; SEQ = 256; REF = [8, 10, 12]; K = 64; L_PROBE = 8
H = m.transformer.h
CTL = {'keep': None}   # None = no intervention; set of (L,h) kept dynamic, rest const
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
def run_condition(blocks, keep, tok, xbar8, Uc, base_dev_energy):
    """returns CE and content retention at L8."""
    CTL['keep'] = keep; CAP8.clear()
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); n += tgt.shape[0]
    CTL['keep'] = None
    X = torch.cat(CAP8, 0); CAP8.clear()
    dev = X - xbar8[tok]; dev = dev - dev.mean(0)
    retained = float(((dev @ Uc)**2).sum()) / base_dev_energy
    return tot/n, retained


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])

    # pass 1 (full model): head means, L8 xbar, content basis, base content energy
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
    tok = torch.cat(idsL, 0)
    cn = torch.zeros(V, device=DEV); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    devsum = None; xbar8 = None
    for Lr in REF:
        X = torch.cat(capR[Lr], 0); capR[Lr] = []
        xb = torch.zeros(V, D, device=DEV); xb.index_add_(0, tok, X)
        xb = xb/cn.clamp_min(1).unsqueeze(1)
        if Lr == L_PROBE: xbar8 = xb.clone()
        dv = X - xb[tok]
        devsum = dv if devsum is None else devsum + dv
        del X
    dev = devsum/len(REF); dev = dev - dev.mean(0)
    _, _, Vt = torch.linalg.svd(dev, full_matrices=False); Uc = Vt[:K].T.contiguous()
    del dev, devsum

    hs = [H[L].attn.c_proj.register_forward_pre_hook(hook(L)) for L in range(18)]
    hc = H[L_PROBE].mlp.register_forward_hook(cap8_hook)
    # base (keep=all): measure baseline content energy with the capture path
    allheads = {(L, h2) for L in range(18) for h2 in range(NH)}
    CTL['keep'] = allheads; CAP8.clear()
    base_ce_tot = 0.0; nn = 0
    for i in range(0, NSEQ, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        base_ce_tot += float(-lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt].sum()); nn += tgt.shape[0]
    CTL['keep'] = None
    Xb = torch.cat(CAP8, 0); CAP8.clear()
    devb = Xb - xbar8[tok]; devb = devb - devb.mean(0)
    base_energy = float(((devb @ Uc)**2).sum()); del Xb, devb
    base_ce = base_ce_tot/nn

    g = torch.Generator().manual_seed(0)
    rand27 = set()
    while len(rand27) < 27:
        rand27.add((int(torch.randint(3, 15, (1,), generator=g)), int(torch.randint(0, NH, (1,), generator=g))))
    gather = {(L, h2) for L in (3, 4, 5) for h2 in range(NH)}
    conds = {
        'front_L0_2': {(L, h2) for L in (0, 1, 2) for h2 in range(NH)},
        'gatherer_L3_5': gather,
        'gatherer_minus_57': gather - {(5, 7)},
        'middle_L6_9': {(L, h2) for L in (6, 7, 8, 9) for h2 in range(NH)},
        'random_27': rand27,
        'none_allconst': set(),
    }
    out = {'base_ce': round(base_ce, 4), 'conditions': {}}
    for name, keep in conds.items():
        c, r = run_condition(blocks, keep, tok, xbar8, Uc, base_energy)
        out['conditions'][name] = {'ce_cost': round(c-base_ce, 4), 'content_retained': round(r, 4)}
        print(f"{name:>20}: CE +{c-base_ce:.4f} | content retained {r:.4f}", flush=True)
    for h in hs: h.remove()
    hc.remove()

    cd = out['conditions']
    ac = cd['none_allconst']['content_retained']
    def gain(name): return cd[name]['content_retained'] - ac
    out['pred_a_seed_in_gatherer'] = bool(cd['gatherer_L3_5']['content_retained'] >= 0.6
                                          and gain('front_L0_2') < 0.5*gain('gatherer_L3_5')
                                          and abs(cd['gatherer_minus_57']['content_retained'] - cd['gatherer_L3_5']['content_retained']) < 0.1)
    out['pred_b_distributed'] = bool(gain('middle_L6_9') >= 0.9*gain('gatherer_L3_5'))
    ranks_ce = sorted(conds, key=lambda n: cd[n]['ce_cost'])
    ranks_ret = sorted(conds, key=lambda n: -cd[n]['content_retained'])
    out['rank_agreement'] = ranks_ce == ranks_ret
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a gatherer-seed {out['pred_a_seed_in_gatherer']} | pred_b distributed {out['pred_b_distributed']} | rank-agree {out['rank_agreement']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
