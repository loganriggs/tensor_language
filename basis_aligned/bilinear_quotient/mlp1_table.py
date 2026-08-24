# mlp1_table: PILOT OF THE MODULE-REPLACEMENT BENCHMARK (user framing 2026-08-24:
# "I want to replace MLP1 entirely — decompose it as a table / clustering thing, also on
# positions, completely characterize the computation, and put a number on it").
# Replace mlp1's ENTIRE output with simple lookup structures and score CE recovery
# against the mean-ablation stake on held-out rows:
#   T1  unigram table: per-token mean output (fallback: global mean for unseen tokens)
#   T2  T1 + position deltas: 4 coarse position-bucket global deltas (the §1233 bits idea)
#   T3  T1 + bigram deltas: mean residual per (prev, tok) bigram seen >= 3x in fit
# Prior anchors: §1088 (L1 token-only held-out 0.93 of the stake), §1183 (every MLP is a
# <= 64-token window fn), L0H3 = exact bigram table.
#
# Registered predictions:
#   pred_a TABLE WORKS: T1 recovery >= 0.85 of the mlp1 stake.
#   pred_b POSITION ADDS LITTLE: T2 - T1 <= 0.03 nats (front grammar is position-light).
#   pred_c BIGRAMS ADD: T3 - T1 >= 0.03 nats (mlp1 reads some local context).
import json, time, sys, torch
import torch.nn.functional as F
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp1_table_results.json'
NFIT = 1920; NR = 480; NBIG = 60000; V = 50257
H = m.transformer.h
CUR = {'toks': None, 'mode': None, 'tab': None, 'posdelta': None,
       'big_id': None, 'big_delta': None, 'mean': None}


def mlp1_hook(mod, args, out):
    if CUR['mode'] is None:
        return out
    if CUR['mode'] == 'mean':
        return CUR['mean'].to(out.dtype).expand_as(out)
    toks = CUR['toks']
    y = CUR['tab'][toks].to(out.dtype)
    if CUR['mode'] in ('t2',):
        B2, T2 = toks.shape
        bucket = (torch.arange(T2, device=toks.device) // 64).clamp(0, 3)
        y = y + CUR['posdelta'][bucket].unsqueeze(0).to(out.dtype)
    if CUR['mode'] in ('t3',):
        prev = torch.cat([toks[:, :1], toks[:, :-1]], 1)
        key = prev.long() * V + toks.long()
        bid = CUR['big_id'].get_tensor(key)
        has = bid >= 0
        delta = torch.zeros_like(y)
        if has.any():
            delta[has] = CUR['big_delta'][bid[has]].to(out.dtype)
        y = y + delta
    return y


class BigMap:
    """int64 key -> id via sorted tensor + searchsorted; -1 if absent."""
    def __init__(self, keys):
        self.sorted, self.order = torch.sort(keys)
        self.n = len(keys)

    def get_tensor(self, q):
        idx = torch.searchsorted(self.sorted, q.reshape(-1))
        idx = idx.clamp(0, self.n - 1)
        hit = self.sorted[idx] == q.reshape(-1)
        out = torch.where(hit, self.order[idx], torch.full_like(idx, -1))
        return out.view(q.shape)


@torch.no_grad()
def fwd(idx):
    CUR['toks'] = idx
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    ROWS = cl.fineweb_rows(NFIT + NR)
    FITR, EVR = ROWS[:NFIT, :T + 1].contiguous(), ROWS[NFIT:, :T + 1].contiguous()

    hk = H[1].mlp.register_forward_hook(mlp1_hook)

    # PASS 1: unigram sums + counts + bigram counts + position-bucket residual (later)
    sums = torch.zeros(V, D, device=DEV)
    cnts = torch.zeros(V, device=DEV)
    bigc = Counter()
    caps = {}
    def cap_hook(mod, args, out):
        caps['out'] = out.detach().float()
        return out
    hk2 = H[1].mlp.register_forward_hook(cap_hook)
    CUR['mode'] = None
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        o = caps['out']
        flat_t = idx.reshape(-1)
        sums.index_add_(0, flat_t, o.reshape(-1, D))
        cnts.index_add_(0, flat_t, torch.ones_like(flat_t, dtype=torch.float))
        prev = torch.cat([idx[:, :1], idx[:, :-1]], 1)
        keys = (prev.long() * V + idx.long()).reshape(-1).tolist()
        bigc.update(keys)
    gmean = sums.sum(0) / cnts.sum()
    tab = torch.where(cnts.unsqueeze(1) > 0, sums / cnts.clamp_min(1).unsqueeze(1),
                      gmean.unsqueeze(0))
    print(f"pass1 done: tokens covered {int((cnts > 0).sum())}", flush=True)
    bigs = [k for k, c in bigc.most_common(NBIG) if c >= 3]
    bkeys = torch.tensor(bigs, dtype=torch.long, device=DEV)
    bmap = BigMap(bkeys)
    print(f"bigrams kept {len(bigs)}", flush=True)

    # PASS 2: bigram residual sums + position-bucket residuals
    bsums = torch.zeros(len(bigs), D, device=DEV)
    bcnts = torch.zeros(len(bigs), device=DEV)
    psums = torch.zeros(4, D, device=DEV)
    pcnts = torch.zeros(4, device=DEV)
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        o = caps['out']
        resid = o - tab[idx]
        prev = torch.cat([idx[:, :1], idx[:, :-1]], 1)
        key = prev.long() * V + idx.long()
        bid = bmap.get_tensor(key)
        hasb = bid >= 0
        if hasb.any():
            bsums.index_add_(0, bid[hasb], resid[hasb])
            bcnts.index_add_(0, bid[hasb], torch.ones(int(hasb.sum()), device=DEV))
        bucket = (torch.arange(idx.shape[1], device=DEV) // 64).clamp(0, 3)
        bb = bucket.unsqueeze(0).expand_as(idx).reshape(-1)
        psums.index_add_(0, bb, resid.reshape(-1, D))
        pcnts.index_add_(0, bb, torch.ones_like(bb, dtype=torch.float))
    hk2.remove()
    bdelta = bsums / bcnts.clamp_min(1).unsqueeze(1)
    pdelta = psums / pcnts.clamp_min(1).unsqueeze(1)
    CUR.update(tab=tab, posdelta=pdelta, big_id=bmap, big_delta=bdelta, mean=gmean)
    print("pass2 done", flush=True)

    def ce_eval(mode):
        CUR['mode'] = mode
        tot = 0.0; n = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV); idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            lse = F.cross_entropy(lo[:, 16:].reshape(-1, lo.shape[-1]),
                                  tg[:, 16:].reshape(-1), reduction='sum')
            tot += float(lse); n += idx.shape[0] * (T - 16)
        return tot / n

    res = {}
    for mode in (None, 'mean', 't1', 't2', 't3'):
        name = mode or 'full'
        res[name] = round(ce_eval(mode), 4)
        print(f"{name}: CE {res[name]}", flush=True)
    hk.remove()
    stake = res['mean'] - res['full']
    rec = {k: round((res['mean'] - res[k]) / max(stake, 1e-6), 4) for k in ('t1', 't2', 't3')}
    pa = rec['t1'] >= 0.85
    pb = (res['t1'] - res['t2']) <= 0.03
    pc = (res['t1'] - res['t3']) >= 0.03
    out = {'n_fit': NFIT, 'n_eval': NR, 'n_bigrams': len(bigs),
           'ce': res, 'stake': round(stake, 4), 'recovery': rec,
           'pred_a_table_works': bool(pa), 'pred_b_position_light': bool(pb),
           'pred_c_bigrams_add': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stake {stake:.4f} | recovery {rec}")
    print(f"pred_a table {pa} | pred_b pos-light {pb} | pred_c bigrams {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
