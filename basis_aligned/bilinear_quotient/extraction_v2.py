# extraction_v2: GOAL-1 RUNG 2 — §1311 showed 7 heads recover only 19% of the induction
# gap because the matchers' input variables are built by the front band collectively.
# Keep the DEPENDENCY CLOSURE: all L0-2 heads (27) + matchers/fetchers + sink = 33 of
# 162; every other head mean-replaced everywhere. Conditions: full | allmean |
# rung1 (7 heads, anchor) | closure (33 heads).
#
# Registered predictions:
#   pred_a INDUCTION SURVIVES WITH CLOSURE: closure ident damage <= 40% of allmean's.
#   pred_b STILL BAD AT PROSE: closure elsewhere damage >= 60% of allmean's.
#   pred_c THE BAND WAS THE DEPENDENCY: closure ident CE >= 1.0 nat better than rung1's.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'extraction_v2_results.json'
NMEAN = 24; NR = 960; W = 128
H = m.transformer.h
CIRCUIT = {(2, 5), (3, 8), (8, 3), (8, 4), (5, 7)}
ANN = {(1, 1), (1, 8)}
FRONT = {(L, h) for L in (0, 1, 2) for h in range(9)}


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def stem(s):
    s = s.strip().lower()
    for suf in ('ing', 'es', 'ed', 's', 'd'):
        if s.endswith(suf) and len(s) - len(suf) > 3:
            return s[:-len(suf)]
    return s


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    stem_id = torch.zeros(50257, dtype=torch.long); smap = {}
    for tok in range(50257):
        try:
            s = stem(enc.decode([tok]))
        except Exception:
            s = f'<{tok}>'
        if s not in smap:
            smap[s] = len(smap)
        stem_id[tok] = smap[s]

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]
    stems = stem_id[toks]
    IDENT = torch.zeros_like(toks, dtype=torch.bool)
    VAR = torch.zeros_like(toks, dtype=torch.bool)
    for b0 in range(0, NR, 64):
        tb = toks[b0:b0 + 64]; gb = tgt[b0:b0 + 64]; sb = stems[b0:b0 + 64]
        q_i = torch.arange(T).view(1, T, 1); p_i = torch.arange(T).view(1, 1, T)
        band = (q_i < p_i) & (q_i >= p_i - W)
        cont = (gb.unsqueeze(1) == gb.unsqueeze(2))
        same_tok = (tb.unsqueeze(1) == tb.unsqueeze(2))
        same_stem = (sb.unsqueeze(1) == sb.unsqueeze(2))
        IDENT[b0:b0 + 64] = (same_tok & cont & band).any(1)
        VAR[b0:b0 + 64] = (same_stem & ~same_tok & cont & band).any(1)
    VAR &= ~IDENT
    IDENT[:, :16] = False; VAR[:, :16] = False
    ELSE = ~IDENT & ~VAR; ELSE[:, :16] = False
    print(f"ident {int(IDENT.sum())} | var {int(VAR.sum())}", flush=True)

    # per-layer means for all 18 layers
    caps = {L: [] for L in range(18)}; hooks = []
    for L in range(18):
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

    MODE = {'keep': None}   # None = full model; else set of (L,h) kept live
    hooks = []
    for L in range(18):
        def mk(L):
            def h(mod, args):
                if MODE['keep'] is None:
                    return args
                y = args[0].clone()
                for hh in range(9):
                    if (L, hh) not in MODE['keep']:
                        y[:, :, hh * 128:(hh + 1) * 128] = ymeans[L][hh * 128:(hh + 1) * 128].to(y.dtype)
                return (y,)
            return h
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))

    NAMES = ('ident', 'var', 'els'); SETS = (IDENT, VAR, ELSE)

    def ce_sets(keep):
        MODE['keep'] = keep
        tots = {k: 0.0 for k in NAMES}; ns = {k: 0 for k in NAMES}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in zip(NAMES, SETS):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    res = {}
    for cname, keep in (('full', None), ('allmean', set()),
                        ('rung1', CIRCUIT | ANN), ('closure', FRONT | CIRCUIT)):
        r = ce_sets(keep)
        res[cname] = {k: round(v, 4) for k, v in r.items()}
        print(f"{cname}: {res[cname]}", flush=True)
    for h in hooks:
        h.remove()

    d = {c: {k: res[c][k] - res['full'][k] for k in NAMES} for c in res if c != 'full'}
    pa = d['closure']['ident'] <= 0.4 * max(d['allmean']['ident'], 1e-4)
    pb = d['closure']['els'] >= 0.6 * d['allmean']['els']
    pc = res['rung1']['ident'] - res['closure']['ident'] >= 1.0
    out = {'n_rows': NR, 'kept_closure': 33,
           'ce': res, 'dmg_vs_full': {c: {k: round(v, 4) for k, v in dd.items()} for c, dd in d.items()},
           'pred_a_survives': bool(pa), 'pred_b_still_bad_prose': bool(pb),
           'pred_c_band_was_dependency': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a survives {pa} | pred_b bad-prose {pb} | pred_c band {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
