# capitalized_committee12: RE-PRICE THE REMOVAL HANDLE WITH ALL TWELVE (§1411: five
# new members at a13/a14). Arms: old-7 (a15-17, = §1398's .208) / new-5 (a13/a14) /
# all-12. Same masks/means machinery, NR=960.
#
# Registered predictions:
#   pred_a all-12 target damage >= .30 (the handle grows with the roster).
#   pred_b still surgical: all-12 elsewhere damage <= 10% of its target damage.
#   pred_c sub-additive: all-12 <= .85x (old-7 + new-5) target damages (§1404
#          redundancy-is-shared expected at the band's own grain).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'capitalized_committee12_results.json'
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
    arms = {}
    for name, pairs in (('old7', OLD7), ('new5', NEW5), ('all12', COMMITTEE)):
        r = ce_all(abl_of(pairs))
        arms[name] = {kk: round(r[kk] - base[kk], 4) for kk in r}
        print(f"{name}: tgt {arms[name]['t']:+.4f} jit {arms[name]['j']:+.4f} "
              f"rand {arms[name]['r']:+.4f} else {arms[name]['e']:+.4f}", flush=True)
    for hk in hooks:
        hk.remove()

    pa = arms['all12']['t'] >= 0.30
    pb = abs(arms['all12']['e']) <= 0.10 * max(arms['all12']['t'], 1e-4)
    parts = arms['old7']['t'] + arms['new5']['t']
    pc = arms['all12']['t'] <= 0.85 * max(parts, 1e-4)
    out = {'n_targets': k, 'n_rows': NR, 'base': {kk: round(v, 4) for kk, v in base.items()},
           'arms': arms, 'sum_parts': round(parts, 4),
           'pred_a_all12_30': bool(pa), 'pred_b_surgical': bool(pb),
           'pred_c_subadditive': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"all12 {arms['all12']['t']} vs parts {parts}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
