# closer_share_spread: ERROR-BAR LEG for the flagship closer table (§1382) — 13.8's
# bracket solo-share on THREE disjoint row draws (every share in the table is
# single-draw). Only 13.8 solo + all-but-13.8 + layer per draw (fast battery).
#
# Registered predictions:
#   pred_a share spread <= +-0.05 across the three draws.
#   pred_b share > 0.90 in every draw.
#   pred_c surgical (else <= 10% of target) in every draw.
import json, time, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'closer_share_spread_results.json'
NMEAN = 24; NR = 1920
L13 = 13
H = m.transformer.h
LAYERS = (13,)
CUR = {'heads': None, 'mean': None}       # heads: set of head idx to ABLATE


def cproj_hook(mod, args):
    if CUR['heads'] is None:
        return None
    y = args[0].clone()
    for h in CUR['heads']:
        y[..., h * 128:(h + 1) * 128] = CUR['mean'][h].to(y.dtype)
    return (y,)


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

    draws = []
    for di, skip in enumerate((0, 160, 320)):
        ROWS = cl.fineweb_rows(NMEAN + NR, skip=skip)[:, :T + 1].contiguous()
        MEANR, EVR = ROWS[:NMEAN], ROWS[NMEAN:]
        # per-head y means
        capsH = []
        hk = H[L13].attn.c_proj.register_forward_pre_hook(
            lambda mod, args: capsH.append(args[0].detach().float().reshape(-1, 9, 128).mean(0)))
        CUR['heads'] = None
        for i in range(0, NMEAN, 4):
            fwd(MEANR[i:i + 4, :-1].to(DEV).contiguous())
        hk.remove()
        CUR['mean'] = torch.stack(capsH).mean(0)
        toks = EVR[:, :-1]; tgt_all = EVR[:, 1:]
        is_open = torch.isin(toks, open_ids); is_close = torch.isin(toks, close_ids)
        depth = torch.zeros_like(toks)
        dr = torch.zeros(toks.shape[0], dtype=torch.long)
        for p in range(toks.shape[1]):
            dr = (dr + is_open[:, p].long() - is_close[:, p].long()).clamp_min(0)
            depth[:, p] = dr
        TARGET = torch.isin(tgt_all, close_ids) & (depth > 0)
        TARGET[:, :64] = False
        ELSE = ~TARGET; ELSE[:, :64] = False
        hook = H[L13].attn.c_proj.register_forward_pre_hook(cproj_hook)

        def ce_pair(abl):
            CUR['heads'] = abl
            st = se = 0.0; nt = ne = 0
            for i in range(0, NR, 8):
                bb = EVR[i:i + 8].to(DEV)
                idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
                lo = fwd(idx).float()
                ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                     reduction='none').view(tg.shape)
                mt = TARGET[i:i + 8].to(DEV); me = ELSE[i:i + 8].to(DEV)
                st += float(ce[mt].sum()); nt += int(mt.sum())
                se += float(ce[me].sum()); ne += int(me.sum())
            return st / max(nt, 1), se / max(ne, 1)

        bt, be = ce_pair(None)
        lt, _ = ce_pair(set(range(9)))
        st_, se_ = ce_pair({8})
        hook.remove()
        layer_dmg = lt - bt
        solo = st_ - bt
        solo_else = se_ - be
        share = solo / max(layer_dmg, 1e-4)
        surgical = abs(solo_else) <= 0.10 * max(solo, 1e-4)
        draws.append({'skip': skip, 'n_targets': int(TARGET.sum()),
                      'layer_dmg': round(layer_dmg, 4), 'solo': round(solo, 4),
                      'share': round(share, 4), 'solo_else': round(solo_else, 4),
                      'surgical': bool(surgical)})
        print(f"draw {di} (skip {skip}): share {share:.4f} | solo {solo:+.4f} of "
              f"{layer_dmg:+.4f} | else {solo_else:+.4f}", flush=True)
        json.dump({'partial': True, 'draws': draws}, open(OUT, 'w'), indent=1)

    shares = [d['share'] for d in draws]
    spread = max(shares) - min(shares)
    pa = spread <= 0.10   # +-0.05
    pb = all(s > 0.90 for s in shares)
    pc = all(d['surgical'] for d in draws)
    out = {'draws': draws, 'share_spread': round(spread, 4),
           'pred_a_spread': bool(pa), 'pred_b_above_090': bool(pb),
           'pred_c_surgical_all': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nshares {shares} | spread {spread:.4f}")
    print(f"pred_a spread {pa} | pred_b >0.90 {pb} | pred_c surgical {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
