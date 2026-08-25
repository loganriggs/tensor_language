# newline_crew_screen: POWER-SCREEN THE REMAINING §1414 CANDIDATES. (1) 16.3 at
# capitalized targets, NR=1920 (it sat at .014 solo in §1397's NR=960 battery — one
# tick under the .015 committee bar; the sharper sample decides membership). (2) The
# second-tier newline candidates 7.2 / 10.2 / 12.6 at newline targets (is the crew
# bigger than 8.2+11.0?). (3) Joint arm: 8.2+11.0+{newline survivors >= .03 clean} vs
# the pair's §1416 joint .2477. skip=5600 rows, NR=1920.
#
# Registered predictions:
#   pred_a 16.3 clears .015 solo on capitalized targets at NR=1920 (13th member).
#   pred_b >= 1 of {7.2, 10.2, 12.6} clears .03 clean on newline targets.
#   pred_c the enlarged crew's joint newline damage >= 1.15x the pair-only joint
#          (measured in the same run; vacuous-guard: scored only if a survivor exists,
#          else FAILED by construction).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_crew_screen_results.json'
NMEAN = 24; NR = 1920
H = m.transformer.h
CANDS = [((16, 3), 'capitalized'), ((7, 2), 'newline'), ((10, 2), 'newline'),
         ((12, 6), 'newline')]
PAIR = {(8, 2), (11, 0)}
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
    nl = set(); cap = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '\n' in d:
            nl.add(tok)
        if len(d) >= 2 and d[0] == ' ' and d[1].isupper() and d[1:].isalpha():
            cap.add(tok)
    IDS = {'newline': torch.tensor(sorted(nl)), 'capitalized': torch.tensor(sorted(cap))}

    ROWS = cl.fineweb_rows(NMEAN + NR, skip=5600)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    need_layers = sorted({L for (L, h), _ in CANDS} | {L for (L, h) in PAIR})
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
    results = {}
    for (L, h), cls in CANDS:
        r = ce_all((L, h))
        d = {c: {kk: round(r[c][kk] - base[c][kk], 4) for kk in 'TJRE'} for c in r}
        td = d[cls]['T']; jd = d[cls]['J']
        clean = jd <= max(0.010, 0.33 * max(td, 1e-4))
        results[f'{L}.{h}'] = {'class': cls, 'dmg': d[cls], 'clean': bool(clean)}
        print(f"{L}.{h}/{cls}: T {td:+.4f} J {jd:+.4f} clean={clean}", flush=True)
        json.dump({'partial': True, 'results': results}, open(OUT, 'w'), indent=1)

    survivors = [(L, h) for (L, h), cls in CANDS
                 if cls == 'newline' and results[f'{L}.{h}']['clean']
                 and results[f'{L}.{h}']['dmg']['T'] >= 0.03]
    rpair = ce_all(set(PAIR))
    pair_joint = rpair['newline']['T'] - base['newline']['T']
    crew_joint = None
    if survivors:
        rcrew = ce_all(set(PAIR) | set(survivors))
        crew_joint = rcrew['newline']['T'] - base['newline']['T']
    for hk in hooks:
        hk.remove()

    r163 = results['16.3']
    pa = r163['clean'] and r163['dmg']['T'] >= 0.015
    pb = len(survivors) >= 1
    pc = bool(survivors) and crew_joint is not None \
        and crew_joint >= 1.15 * max(pair_joint, 1e-4)
    out = {'n': {c: masks[c]['n'] for c in masks}, 'results': results,
           'survivors': [f'{a}.{b}' for a, b in survivors],
           'pair_joint': round(pair_joint, 4),
           'crew_joint': (round(crew_joint, 4) if crew_joint is not None else None),
           'pred_a_163_member': bool(pa), 'pred_b_crew_grows': bool(pb),
           'pred_c_joint_grows': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"survivors {out['survivors']} | pair {pair_joint:.4f} crew {crew_joint}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
