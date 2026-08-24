# stem_induction: natural-text certification of §1307's weights-read prediction — the
# matcher criterion is (approximately) STEM identity, so the copy circuit should fire
# across inflectional variants ("story ... stories") at partial strength. Targets:
# positions p where a DIFFERENT token with the SAME stem occurred at q in the prior 128
# AND tgt[q] == tgt[p] (variant-copy would be correct), with NO exact-token support
# (excluded). Ablate the matcher pair (2.5+3.8, c_proj-slice mean) vs a control pair
# (2.0+3.0); measure damage at variant targets, identical targets, elsewhere.
#
# Stems: decode -> strip -> lower -> remove one suffix of {s, es, ed, ing, d} if len>3.
#
# Registered predictions:
#   pred_a VARIANT INDUCTION EXISTS: matcher ablation damages variant-supported targets
#          >= 0.10 nat with concentration >= 3 over elsewhere.
#   pred_b HALF-STRENGTH BAND: variant damage is 30-80% of identical-supported damage.
#   pred_c CONTROL PAIR FLAT: control ablation at variant targets <= 20% of matcher
#          ablation's.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'stem_induction_results.json'
NMEAN = 24; NR = 960; W = 128
H = m.transformer.h
MATCH = ((2, 5), (3, 8)); CTRL = ((2, 0), (3, 0))


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
    stem_id = torch.zeros(50257, dtype=torch.long)
    smap = {}
    for tok in range(50257):
        try:
            s = stem(enc.decode([tok]))
        except Exception:
            s = f'<{tok}>'
        if s not in smap:
            smap[s] = len(smap)
        stem_id[tok] = smap[s]
    print(f"stems: {len(smap)}", flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt = EVR[:, 1:]
    stems = stem_id[toks]
    VAR = torch.zeros_like(toks, dtype=torch.bool)
    IDENT = torch.zeros_like(toks, dtype=torch.bool)
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
    VAR[:, :16] = False; IDENT[:, :16] = False
    ELSE = ~VAR & ~IDENT; ELSE[:, :16] = False
    print(f"variant targets {int(VAR.sum())} | identical targets {int(IDENT.sum())}", flush=True)

    # per-pair slice means
    caps = {2: [], 3: []}
    hooks = []
    for L in (2, 3):
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
    for L in (2, 3):
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

    NAMES = ('var', 'ident', 'els')
    SETS = (VAR, IDENT, ELSE)

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
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(())
    rm = ce_sets(MATCH)
    rc = ce_sets(CTRL)
    for h in hooks:
        h.remove()
    dm = {k: rm[k] - base[k] for k in NAMES}
    dc = {k: rc[k] - base[k] for k in NAMES}
    conc_var = dm['var'] / max(dm['els'], 1e-4)
    frac = dm['var'] / max(dm['ident'], 1e-4)
    pa = dm['var'] >= 0.10 and conc_var >= 3
    pb = 0.3 <= frac <= 0.8
    pc = dc['var'] <= 0.2 * max(dm['var'], 1e-4)
    out = {'n_rows': NR, 'n_var': int(VAR.sum()), 'n_ident': int(IDENT.sum()),
           'base': {k: round(v, 4) for k, v in base.items()},
           'matcher_dmg': {k: round(v, 4) for k, v in dm.items()},
           'ctrl_dmg': {k: round(v, 4) for k, v in dc.items()},
           'conc_var': round(conc_var, 2), 'var_over_ident': round(frac, 3),
           'pred_a_variant_induction': bool(pa), 'pred_b_half_strength': bool(pb),
           'pred_c_ctrl_flat': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"matcher dmg var {dm['var']:.4f} ident {dm['ident']:.4f} els {dm['els']:.4f} | frac {frac:.3f}")
    print(f"pred_a var-induction {pa} | pred_b half-strength {pb} | pred_c ctrl {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
