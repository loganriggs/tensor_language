# open_quote_screen: POOL CANDIDATE #3 at the §1302 standard. The atlas (§513) flagged
# open_quote -> a10 (+0.272 target vs +0.020 elsewhere, n=90) with a DIRTY control (8.9).
# Clean screen at 1920 rows, whole-layer y-mean ablation, BOTH controls, competitors.
#
# Target definition (registered assumption): next token is an OPENING quote = its decoded
# form starts with a space followed by a quote character (' "', ' \'', unicode open
# quotes) — space-led quotes open in GPT-2's tokenization; bare '"' tokens usually close
# and are excluded. The capability: predicting that quoted speech/citation begins here.
#
# Arms: a10 (candidate), a9, a11, a13, a5 (generalist competitor).
# Registered predictions (verdict withheld if controls dirty):
#   pred_a CONCENTRATION AT SCALE: a10 target damage >= 0.15 nats AND >= 8x elsewhere.
#   pred_b BOTH CONTROLS CLEAN: jitter and random each <= 1.5x.
#   pred_c OWNERSHIP: a10 >= 3x every competitor layer's target damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'open_quote_screen_results.json'
NMEAN = 24; NR = 1920
H = m.transformer.h
LAYERS = (10, 9, 11, 13, 5)
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
    oq = set()
    QCH = ('"', "'", '\u201c', '\u2018')
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if len(d) >= 2 and d[0] == ' ' and d[1] in QCH:
            oq.add(tok)
    oq_ids = torch.tensor(sorted(oq))
    print(f"open-quote token ids: {len(oq)}", flush=True)

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

    # target mask: next tok is an opening quote
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    TARGET = torch.isin(tgt_all, oq_ids)
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

    d13 = dmg[10]
    conc = d13['target'] / max(abs(d13['else']), 1e-4)
    jit_c = d13['jitter'] / max(abs(d13['else']), 1e-4)
    rnd_c = d13['random'] / max(abs(d13['else']), 1e-4)
    pa = d13['target'] >= 0.15 and conc >= 8.0
    pb = jit_c <= 1.5 and rnd_c <= 1.5
    pc = all(d13['target'] >= 3.0 * max(dmg[L]['target'], 1e-4)
             for L in LAYERS if L != 10)
    out = {'n_targets': k, 'n_rows': NR, 'base': {kk: round(v, 4) for kk, v in base.items()},
           'damage': {f'a{L}': dmg[L] for L in LAYERS},
           'a10_concentration': round(conc, 2), 'a10_jitter_conc': round(jit_c, 2),
           'a10_random_conc': round(rnd_c, 2),
           'pred_a_concentration': bool(pa), 'pred_b_controls_clean': bool(pb),
           'pred_c_ownership': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\na10 conc {conc:.1f} | jitter {jit_c:.2f} | random {rnd_c:.2f}")
    print(f"pred_a conc {pa} | pred_b controls {pb} | pred_c ownership {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
