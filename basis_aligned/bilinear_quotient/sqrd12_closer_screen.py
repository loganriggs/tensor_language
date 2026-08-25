# sqrd12_closer_screen: THE BOUNDARY OF THE UNIVERSALITY CLAIM (§1375). sqrd12 = the
# single-branch, row-normalized sibling; §1215-18: the score function decides
# implementation (sqrd12 has NO matcher). Does it still build a closer? Port of the
# §1373 screen.
#
# Registered predictions:
#   pred_a a concentrated close-bracket layer exists (dmg >= 0.20, conc >= 8).
#   pred_b it sits mid-stack (rel depth 0.5-0.85).
#   pred_c both controls clean (<= 1.5) — else verdict withheld.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from tier2_model import load_elriggs
import census_lib as cl

D = 768; T = 256; NL = 12
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'sqrd12_closer_screen_results.json'
NMEAN = 24; NR = 1920
m12, cfg = load_elriggs('sqrd12')
H = m12.transformer.h
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
    x = F.rms_norm(m12.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m12.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(cl.PT + 'census_state_diverse.pt')
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
    V12 = m12.transformer.wte.weight.shape[0]
    ROWS = ROWS.clamp_max(V12 - 1)
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    caps = {L: [] for L in range(NL)}
    hs = []
    for L in range(NL):
        def mk(L):
            def h(mod, args, out):
                y = out[0] if isinstance(out, tuple) else out
                caps[L].append(y.detach().float().mean((0, 1)))
                return out
            return h
        hs.append(H[L].attn.register_forward_hook(mk(L)))
    CUR['abl'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to('cuda').contiguous())
    for h in hs:
        h.remove()
    CUR['mean'] = {L: torch.stack(caps[L]).mean(0) for L in range(NL)}

    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    is_open = torch.isin(toks, open_ids); is_close = torch.isin(toks, close_ids)
    depth = torch.zeros_like(toks)
    dr = torch.zeros(toks.shape[0], dtype=torch.long)
    for p in range(toks.shape[1]):
        dr = (dr + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
        depth[:, p] = dr
    TARGET = torch.isin(tgt_all, close_ids) & (depth > 0)
    TARGET[:, :64] = False
    JIT = torch.zeros_like(TARGET)
    JIT[:, 2:] = TARGET[:, :-2]
    JIT &= ~TARGET
    g = torch.Generator().manual_seed(97)
    sc = torch.rand(TARGET.shape, generator=g)
    sc[TARGET | JIT] = -1.0; sc[:, :64] = -1.0
    k = int(TARGET.sum())
    flat = sc.flatten()
    RAND = torch.zeros_like(flat, dtype=torch.bool)
    RAND[flat.topk(k).indices] = True
    RAND = RAND.view(TARGET.shape)
    ELSE = ~TARGET & ~JIT & ~RAND; ELSE[:, :64] = False
    print(f"targets {k}", flush=True)

    hooks = [H[L].attn.register_forward_hook(mk_hook(L)) for L in range(NL)]

    def ce_all(abl):
        CUR['abl'] = abl
        st = sj = sr = se = 0.0; nt = nj = nr_ = ne = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to('cuda')
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for M, acc in (('t', TARGET), ('j', JIT), ('r', RAND), ('e', ELSE)):
                mm = acc[i:i + 8].to('cuda')
                s_ = float(ce[mm].sum()); n_ = int(mm.sum())
                if M == 't':
                    st += s_; nt += n_
                elif M == 'j':
                    sj += s_; nj += n_
                elif M == 'r':
                    sr += s_; nr_ += n_
                else:
                    se += s_; ne += n_
        return {'target': st / max(nt, 1), 'jitter': sj / max(nj, 1),
                'random': sr / max(nr_, 1), 'else': se / max(ne, 1)}

    base = ce_all(None)
    print(f"base target {base['target']:.3f}", flush=True)
    dmg = {}
    for L in range(NL):
        r = ce_all(L)
        dmg[L] = {kk: round(r[kk] - base[kk], 4) for kk in r}
        print(f"a{L}: t {dmg[L]['target']:+.3f} j {dmg[L]['jitter']:+.3f} "
              f"r {dmg[L]['random']:+.3f} e {dmg[L]['else']:+.3f}", flush=True)
        json.dump({'partial': True, 'damage': {f'a{x}': dmg[x] for x in dmg}},
                  open(OUT, 'w'), indent=1)
    for h in hooks:
        h.remove()

    win = max(dmg, key=lambda L: dmg[L]['target'])
    dw = dmg[win]
    conc = dw['target'] / max(abs(dw['else']), 1e-4)
    jc = dw['jitter'] / max(abs(dw['else']), 1e-4)
    rc = dw['random'] / max(abs(dw['else']), 1e-4)
    pa = dw['target'] >= 0.20 and conc >= 8.0
    pb = 0.5 <= (win / (NL - 1)) <= 0.85
    pc = jc <= 1.5 and rc <= 1.5
    out = {'n_targets': k, 'n_rows': NR, 'damage': {f'a{x}': dmg[x] for x in dmg},
           'winner': f'a{win}', 'winner_rel_depth': round(win / (NL - 1), 3),
           'conc': round(conc, 1), 'jitter_conc': round(jc, 2), 'random_conc': round(rc, 2),
           'pred_a_concentrated': bool(pa), 'pred_b_mid_stack': bool(pb),
           'pred_c_controls_clean': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwinner a{win} (rel depth {win/(NL-1):.2f}) conc {conc:.1f} "
          f"| jitter {jc:.2f} random {rc:.2f}")
    print(f"pred_a conc {pa} | pred_b mid {pb} | pred_c controls {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
