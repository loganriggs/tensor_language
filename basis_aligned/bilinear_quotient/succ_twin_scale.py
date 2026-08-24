# succ_twin_scale: USER CHALLENGE to the §1280/§1281 dormant-twin verdict — "if it's in
# the weights but you didn't see it, can't that mean you just didn't run the right data?
# If you run 10x more data, won't they cover for each other?" Two distinct hypotheses:
#   H_coverage: 14.4 is ACTIVE on natural successor contexts we didn't sample (fix: more
#               and broader data — 2000 natural rows x FOUR lexicons vs §1280's 480 rows
#               x digit lists only; >10x the target count).
#   H_backup:   14.4 only wakes when 8.7 is DEAD (fix: joint-vs-solo ablation, i.e. the
#               interaction term, now at scale and per-lexicon).
# Targets (per lexicon): next token is succ(t) for some same-lexicon token t in the prior
# 128 positions — digits 1-9 ('d','d.','d)'), weekdays (cyclic), months (cyclic), years
# 1500-2098. Conditions: base, ablate 8.7, ablate 14.4, ablate both, control head 8.1.
# Per-head c_proj-slice mean ablation (the §1276/§1280 instrument).
#
# Registered predictions (direction: dormancy + no-backup SURVIVE the scale-up):
#   pred_a DORMANCY HOLDS: pooled 14.4-solo damage <= 0.3x of 8.7-solo (the §1280 bar).
#   pred_b NO BACKUP: pooled interaction (both - h87 - h144) <= 0.2x of 8.7-solo.
#   pred_c ANCHOR: 8.7-solo pooled damage >= 0.10 nat with ELSE damage <= 0.02 (specific).
#   pred_d CONTROL FLAT: |8.1-solo pooled damage| <= 0.2x of 8.7-solo.
# Per-lexicon breakdown reported regardless — a niche where 14.4 is alive falsifies
# pred_a IN THAT NICHE even if the pooled bar passes; report both, hide nothing.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'succ_twin_scale_results.json'
NMEAN = 24; NR = 2000; W = 128
H = m.transformer.h


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def ctx_any(prev, W):
    # ctx[:, p] = any prev in positions [p-W, p-1]
    c = F.pad(prev.float().cumsum(1), (1, 0))
    p = torch.arange(prev.shape[1])
    return (c[:, p] - c[:, (p - W).clamp(min=0)]) > 0


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    WD = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    MO = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December']
    dig = {d: set() for d in range(1, 10)}
    word = {}
    yr = {}
    for tok in range(50257):
        try:
            s = enc.decode([tok]).strip()
        except Exception:
            continue
        for d in range(1, 10):
            if s in (str(d), f"{d}.", f"{d})"):
                dig[d].add(tok)
        if s in WD or s in MO:
            word.setdefault(s, set()).add(tok)
        if len(s) == 4 and s.isascii() and s.isdigit() and 1500 <= int(s) <= 2099:
            yr.setdefault(int(s), set()).add(tok)

    def tt(x):
        return torch.tensor(sorted(x), dtype=torch.long)

    PAIRS = {'digits': [], 'weekdays': [], 'months': [], 'years': []}
    for d in range(1, 9):
        PAIRS['digits'].append((tt(dig[d]), tt(dig[d + 1])))
    for i in range(7):
        a, b = WD[i], WD[(i + 1) % 7]
        if a in word and b in word:
            PAIRS['weekdays'].append((tt(word[a]), tt(word[b])))
    for i in range(12):
        a, b = MO[i], MO[(i + 1) % 12]
        if a in word and b in word:
            PAIRS['months'].append((tt(word[a]), tt(word[b])))
    for y in range(1500, 2098):
        if y in yr and (y + 1) in yr:
            PAIRS['years'].append((tt(yr[y]), tt(yr[y + 1])))
    print({k: len(v) for k, v in PAIRS.items()}, flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    MASKS = {}
    for name, pairs in PAIRS.items():
        M = torch.zeros_like(tgt_all, dtype=torch.bool)
        for prev_t, next_t in pairs:
            is_n = torch.isin(tgt_all, next_t)
            if not is_n.any():
                continue
            prev = torch.isin(toks, prev_t)
            M |= (is_n & ctx_any(prev, W))
        M[:, :64] = False
        MASKS[name] = M
        print(f"{name}: {int(M.sum())} targets", flush=True)
    POOL = MASKS['digits'] | MASKS['weekdays'] | MASKS['months'] | MASKS['years']
    ELSE = ~POOL; ELSE[:, :64] = False
    print(f"pooled targets {int(POOL.sum())} (vs §1280's digit-only 480-row set)", flush=True)

    # head-slice means from MEANR
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

    # ablation hooks: SEL holds set of (layer, head)
    SEL = {'set': set()}
    hooks = []
    for L in (8, 14):
        def mk(L):
            def h(mod, args):
                sel = [hh for (LL, hh) in SEL['set'] if LL == L]
                if not sel:
                    return args
                y = args[0].clone()
                for hh in sel:
                    y[:, :, hh * 128:(hh + 1) * 128] = ymeans[L][hh * 128:(hh + 1) * 128].to(y.dtype)
                return (y,)
            return h
        hooks.append(H[L].attn.c_proj.register_forward_pre_hook(mk(L)))

    NAMES = ['digits', 'weekdays', 'months', 'years', 'pool', 'else']
    SETS = [MASKS['digits'], MASKS['weekdays'], MASKS['months'], MASKS['years'], POOL, ELSE]

    def ce_sets(sel):
        SEL['set'] = sel
        tots = {k: 0.0 for k in NAMES}; ns = {k: 0 for k in NAMES}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in zip(NAMES, SETS):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}, dict(ns)

    base, counts = ce_sets(set())
    print(f"base {base}", flush=True)
    conds = {'h87': {(8, 7)}, 'h144': {(14, 4)}, 'both': {(8, 7), (14, 4)}, 'ctrl81': {(8, 1)}}
    res = {}
    for cname, sel in conds.items():
        r, _ = ce_sets(sel)
        res[cname] = {k: round(r[k] - base[k], 4) for k in NAMES}
        print(f"{cname}: {res[cname]}", flush=True)
    for h in hooks:
        h.remove()

    d87, d144, db, dc = (res[c]['pool'] for c in ('h87', 'h144', 'both', 'ctrl81'))
    inter = db - d87 - d144
    niche_alive = {k: bool(res['h144'][k] > 0.3 * max(res['h87'][k], 1e-4))
                   for k in ('digits', 'weekdays', 'months', 'years')}
    out = {'n_rows': NR, 'counts': {k: counts[k] for k in NAMES},
           'base': {k: round(base[k], 4) for k in NAMES},
           'dmg': res, 'interaction_pool': round(inter, 4),
           'niche_144_alive': niche_alive,
           'pred_a_dormant': bool(d144 <= 0.3 * d87),
           'pred_b_no_backup': bool(inter <= 0.2 * d87),
           'pred_c_anchor': bool(d87 >= 0.10 and abs(res['h87']['else']) <= 0.02),
           'pred_d_ctrl_flat': bool(abs(dc) <= 0.2 * d87),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pool dmg: 8.7 {d87:.4f} | 14.4 {d144:.4f} | both {db:.4f} | interaction {inter:.4f} | ctrl {dc:.4f}")
    print(f"niche_144_alive {niche_alive}")
    print(f"pred_a dormant {out['pred_a_dormant']} | pred_b no-backup {out['pred_b_no_backup']} | "
          f"pred_c anchor {out['pred_c_anchor']} | pred_d ctrl {out['pred_d_ctrl_flat']}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
