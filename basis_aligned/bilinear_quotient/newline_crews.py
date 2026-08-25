# newline_crews: SUB-BAND STRUCTURE OF THE NEWLINE CREW + COMMITTEE-13 REPRICE
# (§1418: crew {7.2, 8.2, 10.2, 11.0, 12.6} joint .617; 16.3 = 13th member). Arms:
# early {7.2, 8.2} / late {10.2, 11.0, 12.6} / all-5 (newline targets), and
# committee-12 / committee-13 (capitalized targets). Same rows (skip=5600, NR=1920).
#
# Registered predictions:
#   pred_a each newline sub-band alone <= .5x the all-5 joint (mutual backups).
#   pred_b all-5 >= 1.3x (early + late) — super-additivity lives BETWEEN sub-bands.
#   pred_c committee-13 >= committee-12 + .01 on capitalized targets.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'newline_crews_results.json'
NMEAN = 24; NR = 1920
H = m.transformer.h
EARLY = {(7, 2), (8, 2)}
LATE = {(10, 2), (11, 0), (12, 6)}
C12 = {(13, 0), (13, 5), (14, 4), (14, 6), (14, 7), (15, 3), (16, 0), (16, 4),
       (16, 5), (17, 0), (17, 1), (17, 2)}
C13 = C12 | {(16, 3)}
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

    need_layers = sorted({L for (L, h) in EARLY | LATE | C13})
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
    arms = {}
    for name, hs in (('early', EARLY), ('late', LATE), ('all5', EARLY | LATE),
                     ('c12', C12), ('c13', C13)):
        r = ce_all(set(hs))
        arms[name] = {c: {kk: round(r[c][kk] - base[c][kk], 4) for kk in 'TJRE'}
                      for c in r}
        print(f"{name}: nl {arms[name]['newline']['T']:+.4f} "
              f"cap {arms[name]['capitalized']['T']:+.4f}", flush=True)
        json.dump({'partial': True, 'arms': arms}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    e, l, a5 = (arms[x]['newline']['T'] for x in ('early', 'late', 'all5'))
    c12d, c13d = arms['c12']['capitalized']['T'], arms['c13']['capitalized']['T']
    pa = e <= 0.5 * a5 and l <= 0.5 * a5
    pb = a5 >= 1.3 * max(e + l, 1e-4)
    pc = c13d >= c12d + 0.01
    out = {'n': {c: masks[c]['n'] for c in masks}, 'arms': arms,
           'newline': {'early': round(e, 4), 'late': round(l, 4), 'all5': round(a5, 4)},
           'capitalized': {'c12': round(c12d, 4), 'c13': round(c13d, 4)},
           'pred_a_subbands_half': bool(pa), 'pred_b_superadditive': bool(pb),
           'pred_c_c13_gain': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"early {e:.4f} late {l:.4f} all5 {a5:.4f} | c12 {c12d:.4f} c13 {c13d:.4f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
