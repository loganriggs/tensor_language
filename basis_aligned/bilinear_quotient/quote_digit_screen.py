# quote_digit_screen: LAST CREDIBLE POOL CANDIDATES (§1414 mine leftovers after the
# punct-class discredit). (1) 11.5 at quote targets with parity split (third close-side
# server under 13.8+10.6?). (2) 14.7 at digit targets (committee member moonlighting
# into the 8.3/8.7 digit family?). (3) Helper-joint 10.6+11.5 on close-side quotes.
# skip=5600 rows, NR=1920. Clean bar: jitter proxy = ELSE controls (parity masks lack
# jitter; ELSE <= .33x target).
#
# Registered predictions:
#   pred_a 11.5 >= .03 on quote targets, clean, with close-side >= open-side.
#   pred_b 14.7 >= .03 on digit targets, clean.
#   pred_c 10.6+11.5 joint close-quote damage >= 1.3x the larger solo.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'quote_digit_screen_results.json'
NMEAN = 24; NR = 1920
H = m.transformer.h
H115 = (11, 5); H147 = (14, 7); H106 = (10, 6)
CUR = {'head': None, 'mean': {}}


def mk_hook(L):
    def hook(mod, args):
        if not CUR['head']:
            return None
        y = None
        for (LL, h) in CUR['head']:
            if LL == L:
                if y is None:
                    y = args[0].clone()
                y[..., h * 128:(h + 1) * 128] = CUR['mean'][L][h].to(y.dtype)
        return None if y is None else (y,)
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
    qu = set(); dg = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '"' in d or "'" in d:
            qu.add(tok)
        if d.strip().isdigit():
            dg.add(tok)
    IDS = {'quote': torch.tensor(sorted(qu)), 'digit': torch.tensor(sorted(dg))}

    ROWS = cl.fineweb_rows(NMEAN + NR, skip=5600)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    need_layers = sorted({H115[0], H147[0], H106[0]})
    caps = {L: [] for L in need_layers}
    hks = [H[L].attn.c_proj.register_forward_pre_hook(
        (lambda LL: lambda mod, args: caps[LL].append(
            args[0].detach().float().reshape(-1, 9, 128).mean(0)))(L))
        for L in need_layers]
    CUR['head'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for hk in hks:
        hk.remove()
    for L in need_layers:
        CUR['mean'][L] = torch.stack(caps[L]).mean(0).to(DEV)

    tgt_all = EVR[:, 1:]; toks = EVR[:, :-1]
    isq = torch.isin(toks, IDS['quote'])
    par = (isq.long().cumsum(1) % 2) == 1
    T_quote = torch.isin(tgt_all, IDS['quote'])
    defs = {'quote': T_quote, 'q_open': T_quote & ~par, 'q_close': T_quote & par,
            'digit': torch.isin(tgt_all, IDS['digit'])}
    masks = {}
    for cls, TARGET in defs.items():
        TARGET = TARGET.clone()
        TARGET[:, :64] = False
        JIT = torch.zeros_like(TARGET)
        JIT[:, 2:] = TARGET[:, :-2]
        JIT &= ~TARGET
        ELSE = ~TARGET & ~JIT; ELSE[:, :64] = False
        masks[cls] = dict(T=TARGET, J=JIT, R=TARGET & False, E=ELSE, n=int(TARGET.sum()))
        print(f"{cls}: n={masks[cls]['n']}", flush=True)

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in need_layers]

    def ce_all(head):
        CUR['head'] = (head if head is None or isinstance(head, set) else {head})
        acc = {c: {k: [0.0, 0] for k in 'TJRE'} for c in masks}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for c in masks:
                for kk in 'TJRE':
                    mm = masks[c][kk][i:i + 8].to(DEV)
                    acc[c][kk][0] += float(ce[mm].sum()); acc[c][kk][1] += int(mm.sum())
        return {c: {kk: acc[c][kk][0] / max(acc[c][kk][1], 1) for kk in 'TJRE'}
                for c in masks}

    base = ce_all(None)
    arms = {}
    for name, hs in (('11.5', {H115}), ('14.7', {H147}), ('10.6', {H106}),
                     ('helpers', {H106, H115})):
        r = ce_all(set(hs))
        arms[name] = {c: {kk: round(r[c][kk] - base[c][kk], 4) for kk in 'TJRE'}
                      for c in r}
        print(f"{name}: " + " ".join(f"{c} {arms[name][c]['T']:+.4f}" for c in arms[name]),
              flush=True)
        json.dump({'partial': True, 'arms': arms}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    a115 = arms['11.5']; a147 = arms['14.7']
    clean115 = abs(a115['quote']['E']) <= 0.33 * max(a115['quote']['T'], 1e-4)
    clean147 = abs(a147['digit']['E']) <= 0.33 * max(a147['digit']['T'], 1e-4)
    pa = a115['quote']['T'] >= 0.03 and clean115 \
        and a115['q_close']['T'] >= a115['q_open']['T']
    pb = a147['digit']['T'] >= 0.03 and clean147
    hj = arms['helpers']['q_close']['T']
    solo_max = max(arms['10.6']['q_close']['T'], arms['11.5']['q_close']['T'])
    pc = hj >= 1.3 * max(solo_max, 1e-4)
    out = {'n': {c: masks[c]['n'] for c in masks}, 'arms': arms,
           'helper_joint_close': round(hj, 4), 'solo_max_close': round(solo_max, 4),
           'pred_a_115_close_server': bool(pa), 'pred_b_147_digit': bool(pb),
           'pred_c_helpers_joint': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"helpers joint {hj:.4f} vs solo max {solo_max:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
