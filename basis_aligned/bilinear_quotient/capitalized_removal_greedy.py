# capitalized_removal_greedy: THE MINIMAL REMOVAL SET (§1412: two crews, mutual
# backups, all-12 = .603). Greedy build of the removal set: each round, test every
# remaining committee head joined to the current set, keep the one with max target
# damage; 6 rounds. Marginals re-measured each round (§1404/§1412: they don't compose).
#
# Registered predictions:
#   pred_a the first 4 picks include >= 1 head from EACH crew (a13-14 and a15-17 —
#          efficient removal must hit both implementations).
#   pred_b the 6-head set reaches >= .40 target damage (2/3 of the full .603).
#   pred_c surgical at 6: else <= 10% of target damage.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'capitalized_removal_greedy_results.json'
NMEAN = 24; NR = 960
H = m.transformer.h
LAYERS = (13, 14, 15, 16, 17)
OLD7 = {(17, 2), (17, 1), (16, 5), (15, 3), (17, 0), (16, 4), (16, 0)}
NEW5 = {(14, 4), (13, 5), (14, 7), (13, 0), (14, 6)}
COMMITTEE = OLD7 | NEW5
CUR = {'abl': None, 'mean': {}}


def mk_hook(L):
    def hook(mod, args):
        if CUR['abl'] is None or L not in CUR['abl'] or not CUR['abl'][L]:
            return None
        y = args[0].clone()
        for h in CUR['abl'][L]:
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
    cap = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if len(d) >= 2 and d[0] == ' ' and d[1].isupper() and d[1:].isalpha():
            cap.add(tok)
    cap_ids = torch.tensor(sorted(cap))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    caps = {L: [] for L in LAYERS}
    hks = [H[L].attn.c_proj.register_forward_pre_hook(
        (lambda LL: lambda mod, args: caps[LL].append(
            args[0].detach().float().reshape(-1, 9, 128).mean(0)))(L)) for L in LAYERS]
    CUR['abl'] = None
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    for hk in hks:
        hk.remove()
    for L in LAYERS:
        CUR['mean'][L] = torch.stack(caps[L]).mean(0)

    tgt_all = EVR[:, 1:]
    TARGET = torch.isin(tgt_all, cap_ids)
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
    print(f"targets {k}", flush=True)

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L)) for L in LAYERS]

    def ce_all(abl):
        CUR['abl'] = abl
        sums = dict(t=0.0, j=0.0, r=0.0, e=0.0); ns = dict(t=0, j=0, r=0, e=0)
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            for key, M in (('t', TARGET), ('j', JIT), ('r', RAND), ('e', ELSE)):
                mm = M[i:i + 8].to(DEV)
                sums[key] += float(ce[mm].sum()); ns[key] += int(mm.sum())
        return {kk: sums[kk] / max(ns[kk], 1) for kk in sums}

    def abl_of(pairs):
        d = {L: set() for L in LAYERS}
        for (L, h) in pairs:
            d[L].add(h)
        return d

    base = ce_all(None)
    chosen = []
    trail = []
    remaining = sorted(COMMITTEE)
    for rnd in range(6):
        best = None
        for pair in remaining:
            r = ce_all(abl_of(chosen + [pair]))
            dmg = r['t'] - base['t']
            if best is None or dmg > best[1]['t'] - base['t']:
                best = (pair, r)
        pair, r = best
        chosen.append(pair)
        remaining.remove(pair)
        step = {kk: round(r[kk] - base[kk], 4) for kk in r}
        trail.append({'pick': f'{pair[0]}.{pair[1]}', 'damage': step})
        print(f"round {rnd + 1}: {pair[0]}.{pair[1]} -> tgt {step['t']:+.4f} "
              f"else {step['e']:+.4f}", flush=True)
        json.dump({'partial': True, 'trail': trail}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    first4 = [tuple(int(x) for x in t['pick'].split('.')) for t in trail[:4]]
    crewA = any(L in (13, 14) for L, _ in first4)
    crewB = any(L in (15, 16, 17) for L, _ in first4)
    final = trail[-1]['damage']
    pa = crewA and crewB
    pb = final['t'] >= 0.40
    pc = abs(final['e']) <= 0.10 * max(final['t'], 1e-4)
    out = {'n_targets': k, 'n_rows': NR, 'trail': trail,
           'chosen': [t['pick'] for t in trail], 'final_damage': final,
           'pred_a_mixes_crews': bool(pa), 'pred_b_6heads_40': bool(pb),
           'pred_c_surgical': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"chosen {out['chosen']} final {final}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
