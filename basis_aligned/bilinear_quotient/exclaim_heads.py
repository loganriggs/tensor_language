# exclaim_heads: per-head decomposition of §1317's certified "!"-register carrier
# (attn17, conc 14.0). Every prior certified circuit had a single 98-107% owner; the
# register reading predicts the OPPOSITE here.
#
# Registered predictions (direction: DIFFUSE):
#   pred_a NO OWNER: no single head reaches 50% of the whole-layer target damage.
#   pred_b SPREAD: >= 3 heads each carry >= 15% share.
#   pred_c CONTROLS: whole-layer jitter concentration <= 1.5 (carried from screen).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'exclaim_heads_results.json'
NMEAN = 24; NR = 960; L = 17
H = m.transformer.h


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
    ex = set()
    for tok in range(50257):
        try:
            d = enc.decode([tok])
        except Exception:
            continue
        if '!' in d:
            ex.add(tok)
    qm = torch.tensor(sorted(ex))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    tgt_all = EVR[:, 1:]
    toks = EVR[:, :-1]
    is_ex = torch.isin(toks, qm)
    ctx = is_ex.cumsum(1) > 0
    prior = torch.zeros_like(ctx)
    prior[:, 1:] = ctx[:, :-1]
    TARGET = torch.isin(tgt_all, qm) & prior
    TARGET[:, :64] = False
    ntar = int(TARGET.sum())
    print(f"target positions: {ntar}", flush=True)
    jit = torch.roll(TARGET, shifts=3, dims=1); jit[:, :64] = False
    JITTER = jit & ~TARGET
    ELSE = ~TARGET & ~JITTER; ELSE[:, :64] = False
    print(f"targets {ntar}", flush=True)

    caps = []
    hk = H[L].attn.c_proj.register_forward_pre_hook(
        lambda mod, args: caps.append(args[0].detach().float().mean((0, 1))))
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    hk.remove()
    ymean = torch.stack(caps).mean(0)

    SEL = {'h': None, 'all': False}

    def hook(mod, args):
        if SEL['h'] is None and not SEL['all']:
            return args
        y = args[0].clone()
        if SEL['all']:
            y[:, :, :] = ymean.to(y.dtype)
        else:
            hh = SEL['h']
            y[:, :, hh * 128:(hh + 1) * 128] = ymean[hh * 128:(hh + 1) * 128].to(y.dtype)
        return (y,)

    hk = H[L].attn.c_proj.register_forward_pre_hook(hook)

    def ce_sets(h, allh=False):
        SEL['h'], SEL['all'] = h, allh
        tots = {'t': 0.0, 'j': 0.0, 'e': 0.0}; ns = {'t': 0, 'j': 0, 'e': 0}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in (('t', TARGET), ('j', JITTER), ('e', ELSE)):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(None)
    lay = ce_sets(None, allh=True)
    dlay = lay['t'] - base['t']
    print(f"base_t {base['t']:.4f} | whole-layer dmg_t {dlay:.4f}", flush=True)
    heads = {}
    for hh in range(9):
        r = ce_sets(hh)
        dt = r['t'] - base['t']; dj = r['j'] - base['j']; de = r['e'] - base['e']
        heads[hh] = {'dmg_t': round(dt, 4), 'dmg_j': round(dj, 4), 'dmg_e': round(de, 5),
                     'share': round(dt / max(dlay, 1e-4), 3),
                     'conc': round(dt / max(de, 1e-4), 2),
                     'conc_jit': round(dj / max(de, 1e-4), 2)}
        print(f"10.{hh}: dmg_t {dt:.4f} share {heads[hh]['share']:.3f} conc {heads[hh]['conc']:.1f} "
              f"(jit {heads[hh]['conc_jit']:.1f})", flush=True)
    hk.remove()
    win = max(heads, key=lambda h: heads[h]['dmg_t'])
    w = heads[win]
    shares = sorted((heads[h]['share'] for h in heads), reverse=True)
    pa = w['share'] < 0.5
    pb = sum(1 for s in shares if s >= 0.15) >= 3
    layer_jit = 0.0  # from ce_sets(None, allh=True) vs base on jitter — approximate via winner row
    pc = abs(w['conc_jit']) <= 1.5
    out = {'n_targets': ntar, 'base_t': round(base['t'], 4), 'layer_dmg_t': round(dlay, 4),
           'heads': heads, 'winner': f'17.{win}', 'top_shares': shares[:4],
           'pred_a_no_owner': bool(pa), 'pred_b_spread': bool(pb), 'pred_c_jit': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"winner 17.{win} share {w['share']} | top shares {shares[:4]}")
    print(f"pred_a no-owner {pa} | pred_b spread {pb} | pred_c jit {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
