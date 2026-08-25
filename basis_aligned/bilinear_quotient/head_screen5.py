# head_screen5: §1302-STANDARD CLEAN SCREENS AT HEAD GRAIN for the §1414 mine's top
# five candidates — 8.2/newline, 11.0/newline, 13.3/punct, 11.7/punct, 10.6/quote.
# Solo OV ablation (y-slice mean) per head, scored on its class's target mask with
# jitter/random/else controls. Target classes: newline = tgt contains '\n'; punct =
# tgt strips to non-alnum non-quote non-bracket (e.g. , ; : - etc, excluding classes
# already owned); quote = tgt contains '"' or "'". Bars fixed in advance: CLEAN means
# jitter <= max(.010, .33x target damage); SURVIVE means clean AND target damage >= .03.
# NR=960, skip=5600 (fresh rows). Assumptions registered.
#
# Registered predictions:
#   pred_a 8.2 newline target damage >= .05 with clean controls.
#   pred_b 10.6 quote target damage >= .04 with clean controls (the §1351 layer-grain
#          demotion was a wash-out, not an absence).
#   pred_c >= 3 of the 5 candidates SURVIVE (clean + >= .03).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'head_screen5_results.json'
NMEAN = 24; NR = 960
H = m.transformer.h
CANDS = [((8, 2), 'newline'), ((11, 0), 'newline'), ((13, 3), 'punct'),
         ((11, 7), 'punct'), ((10, 6), 'quote')]
CUR = {'head': None, 'mean': {}}


def mk_hook(L):
    def hook(mod, args):
        if CUR['head'] is None or CUR['head'][0] != L:
            return None
        h = CUR['head'][1]
        y = args[0].clone()
        y[..., h * 128:(h + 1) * 128] = CUR['mean'][L][h].to(y.dtype)
        return (y,)
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
    nl = set(); pu = set(); qu = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        ds = d.strip()
        if '\n' in d:
            nl.add(tok)
        if '"' in d or "'" in d:
            qu.add(tok)
        elif ds and all(not c.isalnum() for c in ds) and '\n' not in d \
                and not any(c in ds for c in ')]}(['):
            pu.add(tok)
    IDS = {'newline': torch.tensor(sorted(nl)), 'punct': torch.tensor(sorted(pu)),
           'quote': torch.tensor(sorted(qu))}

    ROWS = cl.fineweb_rows(NMEAN + NR, skip=5600)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    need_layers = sorted({L for (L, h), _ in CANDS})
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

    tgt_all = EVR[:, 1:]
    masks = {}
    for cls, ids in IDS.items():
        TARGET = torch.isin(tgt_all, ids)
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
        RAND[flat.topk(min(k, int((flat > 0).sum()))).indices] = True
        RAND = RAND.view(TARGET.shape)
        ELSE = ~TARGET & ~JIT & ~RAND; ELSE[:, :64] = False
        masks[cls] = dict(T=TARGET, J=JIT, R=RAND, E=ELSE, n=k)
        print(f"{cls}: n={k}", flush=True)

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in need_layers]

    def ce_all(head):
        CUR['head'] = head
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
    results = {}
    for (L, h), cls in CANDS:
        r = ce_all((L, h))
        d = {c: {kk: round(r[c][kk] - base[c][kk], 4) for kk in 'TJRE'} for c in r}
        td = d[cls]['T']; jd = d[cls]['J']
        clean = jd <= max(0.010, 0.33 * max(td, 1e-4))
        survive = clean and td >= 0.03
        results[f'{L}.{h}'] = {'class': cls, 'dmg': d[cls], 'all_classes': d,
                               'clean': bool(clean), 'survive': bool(survive)}
        print(f"{L}.{h}/{cls}: T {td:+.4f} J {jd:+.4f} R {d[cls]['R']:+.4f} "
              f"E {d[cls]['E']:+.4f} clean={clean} survive={survive}", flush=True)
        json.dump({'partial': True, 'results': results}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    r82 = results['8.2']; r106 = results['10.6']
    pa = r82['clean'] and r82['dmg']['T'] >= 0.05
    pb = r106['clean'] and r106['dmg']['T'] >= 0.04
    nsurv = sum(1 for v in results.values() if v['survive'])
    pc = nsurv >= 3
    out = {'n': {c: masks[c]['n'] for c in masks}, 'results': results,
           'n_survive': nsurv,
           'pred_a_82_newline': bool(pa), 'pred_b_106_quote': bool(pb),
           'pred_c_3_of_5': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"survivors {nsurv}/5 | pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
