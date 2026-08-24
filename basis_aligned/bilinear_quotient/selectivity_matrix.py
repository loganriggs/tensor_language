# selectivity_matrix: GOAL-2 DELIVERABLE (user 2026-08-24: "remove a circuit without
# affecting other abilities unless they overlap"). Remove each named circuit's private
# heads (c_proj-slice mean ablation) and measure ALL six behaviour target sets + elsewhere
# on the same 960 rows. Declared overlaps: matchers and fetchers BOTH serve ident- and
# variant-copy (same circuit, two stations); everything else is disjoint.
#
# Circuits removed: matchers {2.5,3.8} | fetchers {8.3,8.4} | successor {8.7} |
# comparative {8.1} | delimiter {13.8} | question {10.5} | control {2.0,3.0}.
# Behaviours: ident-copy, variant-copy(stem), successor(digits), than, delim-close, "?".
#
# Registered predictions:
#   pred_a DIAGONAL DOMINANCE: every circuit's damage on its own behaviour(s) >= 5x its
#          damage on every non-overlapping behaviour.
#   pred_b OFF-DIAGONAL QUIET: every non-overlap cell <= 0.05 nats.
#   pred_c CONTROL ROW FLAT: control ablation <= 0.05 on every behaviour.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'selectivity_matrix_results.json'
NMEAN = 24; NR = 960; W = 128
H = m.transformer.h
CIRCUITS = {'matchers': ((2, 5), (3, 8)), 'fetchers': ((8, 3), (8, 4)),
            'successor': ((8, 7),), 'comparative': ((8, 1),),
            'delimiter': ((13, 8),), 'question': ((10, 5),), 'control': ((2, 0), (3, 0))}
OWN = {'matchers': ('ident', 'var'), 'fetchers': ('ident', 'var'),
       'successor': ('succ',), 'comparative': ('than',),
       'delimiter': ('delim',), 'question': ('qmark',), 'control': ()}
LAYERS = sorted({L for hs in CIRCUITS.values() for (L, _) in hs})


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
    dig = {d: set() for d in range(1, 10)}
    qm = set(); sent_end = set(); wh = set(); than = set(); comp = set(); dl = set()
    WH = ['Who', 'What', 'When', 'Where', 'Why', 'How', 'Which', 'Is', 'Are', 'Do', 'Does',
          'Did', 'Can', 'Could', 'Will', 'Would', 'Should']
    COMP = ['bigger', 'smaller', 'better', 'worse', 'larger', 'greater', 'higher', 'lower',
            'faster', 'slower', 'older', 'younger', 'stronger', 'weaker', 'easier', 'harder',
            'longer', 'shorter', 'cheaper', 'richer', 'more', 'less', 'fewer', 'rather']
    for tok in range(50257):
        try:
            dec = enc.decode([tok])
        except Exception:
            stem_id[tok] = -1
            continue
        s = stem(dec)
        if s not in smap:
            smap[s] = len(smap)
        stem_id[tok] = smap[s]
        ds = dec.strip()
        for d in range(1, 10):
            if ds in (str(d), f"{d}.", f"{d})"):
                dig[d].add(tok)
        if '?' in dec:
            qm.add(tok)
        if any(c in dec for c in '.!?'):
            sent_end.add(tok)
        if ds in WH:
            wh.add(tok)
        if ds.lower() == 'than':
            than.add(tok)
        if ds.lower() in COMP:
            comp.add(tok)
        if any(c in dec for c in ['"', '(', '[', '{', ')', ']', '}']):
            dl.add(tok)
    tt = lambda x: torch.tensor(sorted(x))
    qm_t, se_t, wh_t, than_t, comp_t, dl_t = tt(qm), tt(sent_end), tt(wh), tt(than), tt(comp), tt(dl)
    dig = {d: tt(v) for d, v in dig.items()}

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]

    # copy masks (ident + variant)
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

    def ctx_any(prev, Wn):
        c = F.pad(prev.float().cumsum(1), (1, 0))
        p = torch.arange(prev.shape[1])
        return (c[:, p] - c[:, (p - Wn).clamp(min=0)]) > 0

    SUCC = torch.zeros_like(toks, dtype=torch.bool)
    for d in range(1, 9):
        SUCC |= torch.isin(tgt, dig[d + 1]) & ctx_any(torch.isin(toks, dig[d]), 128)

    is_comp = torch.isin(toks, comp_t)
    B2, T2 = toks.shape
    OPENER = torch.full_like(toks, -1); op = torch.full((B2,), -1, dtype=torch.long)
    for p in range(T2):
        op = torch.where(is_comp[:, p], torch.full_like(op, p), op)
        op = torch.where((op >= 0) & (p - op > 20), torch.full_like(op, -1), op)
        OPENER[:, p] = op
    dist = torch.arange(T2).view(1, -1) - OPENER
    THAN = torch.isin(tgt, than_t) & (OPENER >= 8) & (dist >= 2) & (dist <= 20)

    DELIM = torch.isin(tgt, dl_t) & ctx_any(torch.isin(toks, dl_t), 64)

    is_end = torch.isin(toks, se_t); is_wh = torch.isin(toks, wh_t)
    state = torch.zeros(B2, dtype=torch.bool)
    QS = torch.zeros_like(toks, dtype=torch.bool)
    rec = torch.full((B2,), 99, dtype=torch.long)
    for p in range(T2):
        opn = is_wh[:, p] & (rec <= 2)
        state = torch.where(is_end[:, p], torch.zeros_like(state), state | opn)
        QS[:, p] = state
        rec = torch.where(is_end[:, p], torch.zeros_like(rec), rec + 1)
    QMARK = torch.isin(tgt, qm_t) & QS

    MASKS = {'ident': IDENT, 'var': VAR, 'succ': SUCC, 'than': THAN,
             'delim': DELIM, 'qmark': QMARK}
    union = torch.zeros_like(toks, dtype=torch.bool)
    for k in MASKS:
        MASKS[k][:, :64] = False
        union |= MASKS[k]
    MASKS['els'] = ~union
    MASKS['els'][:, :64] = False
    for k, v in MASKS.items():
        print(f"{k}: {int(v.sum())}", flush=True)

    # per-layer slice means
    caps = {L: [] for L in LAYERS}; hooks = []
    for L in LAYERS:
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

    SEL = {'set': ()}
    hooks = []
    for L in LAYERS:
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

    NAMES = list(MASKS.keys())

    def ce_sets(sel):
        SEL['set'] = sel
        tots = {k: 0.0 for k in NAMES}; ns = {k: 0 for k in NAMES}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name in NAMES:
                mm = MASKS[name][i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(())
    print(f"base {base}", flush=True)
    MAT = {}
    for cname, heads in CIRCUITS.items():
        r = ce_sets(heads)
        MAT[cname] = {k: round(r[k] - base[k], 4) for k in NAMES}
        print(f"{cname}: {MAT[cname]}", flush=True)
    for h in hooks:
        h.remove()

    beh = [k for k in NAMES if k != 'els']
    viol_a = []; viol_b = []
    for c in CIRCUITS:
        if c == 'control':
            continue
        own = OWN[c]; others = [b for b in beh if b not in own]
        own_min = min(MAT[c][b] for b in own)
        for b in others:
            if MAT[c][b] > 0.05:
                viol_b.append((c, b, MAT[c][b]))
            if own_min < 5 * max(MAT[c][b], 1e-4):
                viol_a.append((c, b, MAT[c][b]))
    pa = len(viol_a) == 0
    pb = len(viol_b) == 0
    pc = all(MAT['control'][b] <= 0.05 for b in beh)
    out = {'n_rows': NR, 'counts': {k: int(MASKS[k].sum()) for k in NAMES},
           'base': {k: round(v, 4) for k, v in base.items()}, 'matrix': MAT,
           'violations_a': viol_a[:20], 'violations_b': viol_b[:20],
           'pred_a_diagonal': bool(pa), 'pred_b_offdiag_quiet': bool(pb),
           'pred_c_control_flat': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a diagonal {pa} | pred_b offdiag {pb} | pred_c control {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
