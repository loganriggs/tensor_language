# close_bracket_screen: OPEN THE TOP CANDIDATE FROM THE REGISTRY POOL at the §1302
# standard. The behaviour atlas (§513) flagged close_bracket -> a13 as the largest
# concentrated effect ever measured here (+0.694 nats at target vs +0.015 elsewhere) —
# but at n=30 and with the atlas's biased ratio denominator. This is the clean screen:
# 1920 rows, whole-layer y-mean ablation, BOTH controls, competitor layers.
#
# Targets: positions whose NEXT token contains ')' AND an unmatched '(' occurred within
# the previous 64 tokens (the capability = closing an open bracket). Elsewhere = all
# other positions past the burn-in.
#
# Arms: ablate (per-layer output -> batch mean) each of a12, a13, a14, a17, and a5
# (random-ish competitor far from the candidates) — one at a time, all positions.
# Controls for a13: JITTER (damage at target positions shifted +2 — same local text,
# wrong position) and RANDOM (count-matched random position set).
#
# Registered predictions (§1302 evaluability standard — verdict withheld if controls dirty):
#   pred_a CONCENTRATION AT SCALE: a13's target damage >= 0.30 nats AND >= 10x its
#          elsewhere damage.
#   pred_b BOTH CONTROLS CLEAN: jitter and random "concentration" each <= 1.5x
#          (damage at control positions / elsewhere damage).
#   pred_c OWNERSHIP: a13's target damage >= 3x every competitor layer's target damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'close_bracket_screen_results.json'
NMEAN = 24; NR = 1920
H = m.transformer.h
LAYERS = (12, 13, 14, 17, 5)
CUR = {'abl': None, 'mean': None}


def mk_hook(L):
    def hook(mod, args, out):
        if CUR['abl'] == L:
            y = out[0] if isinstance(out, tuple) else out
            rep = CUR['mean'][L].to(y.dtype).expand_as(y)
            return (rep,) + tuple(out[1:]) if isinstance(out, tuple) else rep
        return out
    return hook


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
    close_t = set(); open_t = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if ')' in d:
            close_t.add(tok)
        if '(' in d:
            open_t.add(tok)
    close_ids = torch.tensor(sorted(close_t)); open_ids = torch.tensor(sorted(open_t))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # per-layer attn output means from MEANR
    caps = {L: [] for L in LAYERS}
    hs = []
    for L in LAYERS:
        def mk(L):
            def h(mod, args, out):
                y = out[0] if isinstance(out, tuple) else out
                caps[L].append(y.detach().float().mean((0, 1)))
                return out
            return h
        hs.append(H[L].attn.register_forward_hook(mk(L)))
    CUR['abl'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for h in hs:
        h.remove()
    CUR['mean'] = {L: torch.stack(caps[L]).mean(0) for L in LAYERS}

    # target mask: next tok closes AND unmatched open within 64 back
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    is_open = torch.isin(toks, open_ids); is_close = torch.isin(toks, close_ids)
    depth = torch.zeros_like(toks)
    d_run = torch.zeros(toks.shape[0], dtype=torch.long)
    for p in range(toks.shape[1]):
        d_run = (d_run + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
        depth[:, p] = d_run
    TARGET = torch.isin(tgt_all, close_ids) & (depth > 0)
    TARGET[:, :64] = False
    # jitter control: target positions shifted +2 (clamped), excluding real targets
    JIT = torch.zeros_like(TARGET)
    JIT[:, 2:] = TARGET[:, :-2]
    JIT &= ~TARGET
    # random control: count-matched draw from non-target positions
    g = torch.Generator().manual_seed(97)
    scores = torch.rand(TARGET.shape, generator=g)
    scores[TARGET | JIT] = -1.0; scores[:, :64] = -1.0
    k = int(TARGET.sum())
    flat = scores.flatten()
    idx_top = flat.topk(k).indices
    RAND = torch.zeros_like(flat, dtype=torch.bool); RAND[idx_top] = True
    RAND = RAND.view(TARGET.shape)
    ELSE = ~TARGET & ~JIT & ~RAND; ELSE[:, :64] = False
    print(f"targets {k} | jitter {int(JIT.sum())} | rand {int(RAND.sum())}", flush=True)

    hooks = [H[L].attn.register_forward_hook(mk_hook(L)) for L in LAYERS]

    def ce_all(abl):
        CUR['abl'] = abl
        outs = {}
        st = sj = sr = se = 0.0; nt = nj = nr_ = ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for M, acc in (('t', TARGET), ('j', JIT), ('r', RAND), ('e', ELSE)):
                mm = acc[i:i + 8].to(DEV)
                s = float(ce[mm].sum()); n = int(mm.sum())
                if M == 't':
                    st += s; nt += n
                elif M == 'j':
                    sj += s; nj += n
                elif M == 'r':
                    sr += s; nr_ += n
                else:
                    se += s; ne += n
        return {'target': st / max(nt, 1), 'jitter': sj / max(nj, 1),
                'random': sr / max(nr_, 1), 'else': se / max(ne, 1)}

    base = ce_all(None)
    print(f"base: {{k: round(v, 4) for k, v in base.items()}}", flush=True)
    dmg = {}
    for L in LAYERS:
        r = ce_all(L)
        dmg[L] = {k: round(r[k] - base[k], 4) for k in r}
        print(f"a{L}: dmg target {dmg[L]['target']:+.4f} | jit {dmg[L]['jitter']:+.4f} "
              f"| rand {dmg[L]['random']:+.4f} | else {dmg[L]['else']:+.4f}", flush=True)
    for h in hooks:
        h.remove()

    d13 = dmg[13]
    conc = d13['target'] / max(abs(d13['else']), 1e-4)
    jit_c = d13['jitter'] / max(abs(d13['else']), 1e-4)
    rnd_c = d13['random'] / max(abs(d13['else']), 1e-4)
    pa = d13['target'] >= 0.30 and conc >= 10.0
    pb = jit_c <= 1.5 and rnd_c <= 1.5
    pc = all(d13['target'] >= 3.0 * max(dmg[L]['target'], 1e-4)
             for L in LAYERS if L != 13)
    out = {'n_targets': k, 'n_rows': NR, 'base': {kk: round(v, 4) for kk, v in base.items()},
           'damage': {f'a{L}': dmg[L] for L in LAYERS},
           'a13_concentration': round(conc, 2), 'a13_jitter_conc': round(jit_c, 2),
           'a13_random_conc': round(rnd_c, 2),
           'pred_a_concentration': bool(pa), 'pred_b_controls_clean': bool(pb),
           'pred_c_ownership': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\na13 conc {conc:.1f} | jitter {jit_c:.2f} | random {rnd_c:.2f}")
    print(f"pred_a conc {pa} | pred_b controls {pb} | pred_c ownership {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
