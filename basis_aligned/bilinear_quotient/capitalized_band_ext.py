# capitalized_band_ext: DOES THE COMMITTEE EXTEND TO a13/a14? (§1410: the damage mine
# found capitalized carrying 24%/31% of a13/a14 damage mass — the band is deeper than
# the §1339 screen certified.) §1397-style battery: 18 heads (L13/14 x 9) solo + keep
# at capitalized targets, same masks/means machinery.
#
# Registered predictions:
#   pred_a a14 layer damage >= .04 on capitalized targets (comparable to a15's .046).
#   pred_b >= 1 head in a13/a14 clears .015 solo (committee-grade member exists).
#   pred_c controls clean at the top layer (jitter <= .33x target damage).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'capitalized_band_ext_results.json'
NMEAN = 24; NR = 960
H = m.transformer.h
LAYERS = (13, 14)
CUR = {'abl': None, 'mean': {}}   # abl: {L: set(head idx)} to y-mean


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
    print(f"cap vocab {len(cap)}", flush=True)

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]

    # per-layer y means (c_proj input) over MEANR
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

    base = ce_all(None)
    layer_dmg = {}
    for L in LAYERS:
        r = ce_all({L: set(range(9))})
        layer_dmg[L] = {kk: round(r[kk] - base[kk], 4) for kk in r}
        print(f"a{L} full: tgt {layer_dmg[L]['t']:+.4f} jit {layer_dmg[L]['j']:+.4f} "
              f"rand {layer_dmg[L]['r']:+.4f} else {layer_dmg[L]['e']:+.4f}", flush=True)

    solo = {}; keep = {}
    for L in LAYERS:
        for h in range(9):
            r = ce_all({L: {h}})
            solo[(L, h)] = {kk: round(r[kk] - base[kk], 4) for kk in r}
            r2 = ce_all({L: set(range(9)) - {h}})
            keep[(L, h)] = {kk: round(r2[kk] - base[kk], 4) for kk in r2}
            print(f"{L}.{h}: solo {solo[(L, h)]['t']:+.4f} (else {solo[(L, h)]['e']:+.4f})"
                  f" | all-but {keep[(L, h)]['t']:+.4f}", flush=True)
            json.dump({'partial': True,
                       'solo': {f'{a}.{b}': v for (a, b), v in solo.items()},
                       'keep': {f'{a}.{b}': v for (a, b), v in keep.items()}},
                      open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    per_layer_top = {}
    for L in LAYERS:
        hh = max(range(9), key=lambda h: solo[(L, h)]['t'])
        share = solo[(L, hh)]['t'] / max(layer_dmg[L]['t'], 1e-4)
        per_layer_top[L] = {'head': hh, 'solo': solo[(L, hh)]['t'], 'share': round(share, 4)}
    pa = layer_dmg[14]['t'] >= 0.04
    pb = any(solo[x]['t'] >= 0.015 for x in solo)
    topL = max(LAYERS, key=lambda L: layer_dmg[L]['t'])
    pc = layer_dmg[topL]['j'] <= 0.33 * max(layer_dmg[topL]['t'], 1e-4)
    out = {'n_targets': k, 'n_rows': NR, 'base': {kk: round(v, 4) for kk, v in base.items()},
           'layer_dmg': {str(L): layer_dmg[L] for L in LAYERS},
           'solo': {f'{a}.{b}': v for (a, b), v in solo.items()},
           'keep': {f'{a}.{b}': v for (a, b), v in keep.items()},
           'per_layer_top': {str(L): per_layer_top[L] for L in LAYERS},
           'pred_a_a14_04': bool(pa), 'pred_b_member_exists': bool(pb),
           'pred_c_controls_clean': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"per-layer top: " + " ".join(f"a{L}={per_layer_top[L]}" for L in LAYERS))
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
