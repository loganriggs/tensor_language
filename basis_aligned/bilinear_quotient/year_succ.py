# year_succ: YEAR SUCCESSION — the dormant-twin discriminator (§1280 hypothesis i) and a
# new successor context in one. Many years 1900-2030 are single GPT-2 tokens; target =
# positions whose next token is year Y+1 with year Y in the prior 128.
#
# Registered predictions:
#   pred_a THE GENERAL SUCCESSOR COVERS YEARS: 8.7 mean-ablation concentration >= 3 at
#          year-successor targets.
#   pred_b THE TWIN STAYS DORMANT (bet): 14.4's concentration <= 1.5 there (hypothesis i
#          of §1280 — "14.4 wakes on multi-token/numeric contexts" — REFUTED if this holds;
#          if instead >= 3, the twin's niche is found; logged either way).
#   pred_c WEIGHTS ENCODE YEARS IN BOTH: 8.7's AND 14.4's maps rank Y+1 in the top-3 of
#          {Y-2..Y+2} candidate sets for >= 60% of testable years (the §1279 twin symmetry
#          should hold in weights even if causally asymmetric).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'year_succ_results.json'
NMEAN = 24; NR = 480
H = m.transformer.h


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    yr = {}
    for tok in range(50257):
        try:
            s = enc.decode([tok]).strip()
        except Exception:
            continue
        if s.isascii() and s.isdigit() and len(s) == 4 and 1900 <= int(s) <= 2030:
            yr.setdefault(int(s), []).append(tok)
    years = sorted(yr)
    print(f"single-token years: {len(years)} ({years[0]}..{years[-1]})", flush=True)
    yrt = {y: torch.tensor(v) for y, v in yr.items()}

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    tgt_all = EVR[:, 1:]
    TGT = torch.zeros_like(tgt_all, dtype=torch.bool)
    for y in years:
        if y + 1 not in yrt:
            continue
        is_s = torch.isin(tgt_all, yrt[y + 1])
        prev = torch.isin(EVR[:, :-1], yrt[y])
        ctx = torch.zeros_like(prev)
        for w in range(1, 129):
            sh = torch.zeros_like(prev)
            sh[:, w:] = prev[:, :-w]
            ctx |= sh
        TGT |= (is_s & ctx)
    TGT[:, :64] = False
    ELSE = ~TGT; ELSE[:, :64] = False
    ntar = int(TGT.sum())
    print(f"year-successor targets: {ntar}", flush=True)

    caps = {8: [], 14: []}
    hooks = []
    for L in (8, 14):
        def mk(L):
            def h(mod, args):
                caps[L].append(args[0].detach().float().mean((0, 1)))
            return h
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for h in hooks:
        h.remove()
    ymeans = {L: torch.stack(v).mean(0) for L, v in caps.items()}

    SEL = {'set': set()}
    HEADIDX = {8: 7, 14: 4}
    hooks = []
    for L in (8, 14):
        def mk(L):
            def h(mod, args):
                if L in SEL['set']:
                    y = args[0].clone()
                    hh = HEADIDX[L]
                    y[:, :, hh * 128:(hh + 1) * 128] = ymeans[L][hh * 128:(hh + 1) * 128].to(y.dtype)
                    return (y,)
                return args
            return h
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))

    def ce_sets(sel):
        SEL['set'] = sel
        tots = {'t': 0.0, 'e': 0.0}; ns = {'t': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TGT), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(set())
    r87 = ce_sets({8}); r144 = ce_sets({14})
    for h in hooks:
        h.remove()
    c87 = (r87['t'] - base['t']) / max(r87['e'] - base['e'], 1e-4)
    c144 = (r144['t'] - base['t']) / max(r144['e'] - base['e'], 1e-4)
    print(f"conc 8.7 {c87:.2f} (dmg {r87['t']-base['t']:.4f}) | 14.4 {c144:.2f} (dmg {r144['t']-base['t']:.4f})", flush=True)

    # weights: year successor in both maps
    at0 = H[0].attn
    W_u = m.lm_head.weight.float()
    def head_map(L, h, toks):
        atL = H[L].attn; lam = float(atL.lamb)
        x = F.rms_norm(m.transformer.wte(toks), (D,))
        vL = atL.c_v(x).view(-1, 9, 128)[:, h]
        v0 = at0.c_v(x).view(-1, 9, 128)[:, h]
        vv = (1 - lam) * vL + lam * v0
        y = torch.zeros(vv.shape[0], 9, 128, device=DEV, dtype=vv.dtype)
        y[:, h] = vv
        return (atL.c_proj(y.reshape(-1, D)).float() @ W_u.T).mean(0)
    def wt_frac(L, h):
        ok = 0; n = 0
        for y in years:
            cands = [y + o for o in (-2, -1, 0, 1, 2) if (y + o) in yrt]
            if (y + 1) not in yrt or len(cands) < 4:
                continue
            lg = head_map(L, h, yrt[y].to(DEV))
            means = {c: float(lg[yrt[c].to(DEV)].mean()) for c in cands}
            order = sorted(means.values(), reverse=True)
            rank = order.index(means[y + 1]) + 1
            ok += int(rank <= 3); n += 1
        return ok / max(n, 1), n
    f87, n87 = wt_frac(8, 7)
    f144, _ = wt_frac(14, 4)
    print(f"weights top-3 frac: 8.7 {f87:.2f} | 14.4 {f144:.2f} (n={n87})", flush=True)

    out = {'n_targets': ntar, 'base_t': round(base['t'], 4),
           'conc': {'h87': round(c87, 2), 'h144': round(c144, 2)},
           'dmg': {'h87': round(r87['t'] - base['t'], 4), 'h144': round(r144['t'] - base['t'], 4)},
           'weights_top3_frac': {'h87': round(f87, 3), 'h144': round(f144, 3), 'n_years': n87},
           'pred_a_87_covers_years': bool(c87 >= 3),
           'pred_b_twin_dormant': bool(c144 <= 1.5),
           'pred_c_weights_both': bool(f87 >= 0.6 and f144 >= 0.6),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {out['pred_a_87_covers_years']} | pred_b {out['pred_b_twin_dormant']} | pred_c {out['pred_c_weights_both']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
