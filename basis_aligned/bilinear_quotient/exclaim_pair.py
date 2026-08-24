# exclaim_pair: §1318 pair structure — 17.2/17.3 carry 91% of the "!"-register effect.
# Solo/solo/joint c_proj-slice mean ablation + burst/sustained target split (previous "!"
# within 20 tokens vs farther back).
#
# Registered predictions:
#   pred_a ADDITIVE: joint damage within 20% of the solo-sum (complementary specialists,
#          not redundant coverage — the anti-1.1/1.8 pattern, from §1318 share arithmetic).
#   pred_b BOTH CONCENTRATED: each solo elsewhere damage <= 10% of its target damage.
#   pred_c DIVISION OF LABOR: the burst/sustained damage ratio differs >= 2x between the
#          heads (direction: 17.3 = burst, 17.2 = sustained).
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'exclaim_pair_results.json'
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
    ex_t = torch.tensor(sorted(ex))

    ROWS = cl.fineweb_rows(NMEAN + NR)[:, :T + 1].contiguous()
    MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
    toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
    is_ex = torch.isin(toks, ex_t)
    ctx_all = is_ex.cumsum(1) > 0
    prior = torch.zeros_like(ctx_all); prior[:, 1:] = ctx_all[:, :-1]
    TGT = torch.isin(tgt_all, ex_t) & prior
    TGT[:, :64] = False
    c = F.pad(is_ex.float().cumsum(1), (1, 0))
    p = torch.arange(toks.shape[1])
    recent = (c[:, p] - c[:, (p - 20).clamp(min=0)]) > 0
    BURST = TGT & recent
    SUST = TGT & ~recent
    ELSE = ~TGT; ELSE[:, :64] = False
    print(f"targets {int(TGT.sum())} (burst {int(BURST.sum())} sust {int(SUST.sum())})", flush=True)

    caps = []
    hk = H[L].attn.c_proj.register_forward_pre_hook(
        lambda mod, args: caps.append(args[0].detach().float().mean((0, 1))))
    for i in range(0, NMEAN, 4):
        fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
    hk.remove()
    ymean = torch.stack(caps).mean(0)
    SEL = {'hs': ()}

    def hook(mod, args):
        if not SEL['hs']:
            return args
        y = args[0].clone()
        for hh in SEL['hs']:
            y[:, :, hh * 128:(hh + 1) * 128] = ymean[hh * 128:(hh + 1) * 128].to(y.dtype)
        return (y,)

    hk = H[L].attn.c_proj.register_forward_pre_hook(hook)
    NAMES = ('t', 'b', 's', 'e'); SETS = (TGT, BURST, SUST, ELSE)

    def ce_sets(hs):
        SEL['hs'] = hs
        tots = {k: 0.0 for k in NAMES}; ns = {k: 0 for k in NAMES}
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1), reduction='none').view(tg.shape)
            for name, mask in zip(NAMES, SETS):
                mm = mask[i:i + 8].to(DEV)
                tots[name] += float(lse[mm].sum()); ns[name] += int(mm.sum())
        return {k: tots[k] / max(ns[k], 1) for k in tots}

    base = ce_sets(())
    r2 = ce_sets((2,))
    r3 = ce_sets((3,))
    rj = ce_sets((2, 3))
    hk.remove()
    d2 = {k: r2[k] - base[k] for k in NAMES}
    d3 = {k: r3[k] - base[k] for k in NAMES}
    dj = {k: rj[k] - base[k] for k in NAMES}
    ssum = d2['t'] + d3['t']
    pa = abs(dj['t'] - ssum) <= 0.2 * max(ssum, 1e-4)
    pb = (d2['e'] <= 0.1 * max(d2['t'], 1e-4)) and (d3['e'] <= 0.1 * max(d3['t'], 1e-4))
    rat2 = d2['b'] / max(d2['s'], 1e-4)
    rat3 = d3['b'] / max(d3['s'], 1e-4)
    pc = max(rat2, rat3) >= 2 * max(min(rat2, rat3), 1e-4)
    out = {'n_targets': int(TGT.sum()), 'base_t': round(base['t'], 4),
           'solo_17_2': {k: round(v, 4) for k, v in d2.items()},
           'solo_17_3': {k: round(v, 4) for k, v in d3.items()},
           'joint': {k: round(v, 4) for k, v in dj.items()},
           'solo_sum_t': round(ssum, 4),
           'burst_sust_ratio': {'17.2': round(rat2, 2), '17.3': round(rat3, 2)},
           'pred_a_additive': bool(pa), 'pred_b_concentrated': bool(pb),
           'pred_c_division': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"solo 17.2 {d2['t']:.4f} | solo 17.3 {d3['t']:.4f} | joint {dj['t']:.4f} (sum {ssum:.4f})")
    print(f"burst/sust: 17.2 {rat2:.2f} | 17.3 {rat3:.2f}")
    print(f"pred_a additive {pa} | pred_b conc {pb} | pred_c division {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
