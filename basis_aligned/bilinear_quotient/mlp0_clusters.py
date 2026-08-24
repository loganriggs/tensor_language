# mlp0_clusters: MODULE LADDER, second entry — the same simplicity curve for MLP0 (the
# §387 identity-code generator; front grammar machine). Same instrument as mlp1_clusters:
# per-token mean output table (ceiling) + k-means categories K=1..1024, CE recovery vs
# the mean-ablation stake on held-out rows.
#
# Registered predictions (curve-shape comparison to mlp1 is the point):
#   pred_a BIG STAKE: mlp0's stake >= 3 nats.
#   pred_b TABLE CEILING HIGH: table recovery >= 0.90 (token-driven like mlp1).
#   pred_c SAME LOG-LINEAR SHAPE: K=16 recovery in [0.25, 0.55] (no categorical elbow
#          here either — token-resolved value is a FRONT-MODULE property, not an mlp1
#          quirk).
import json, time, sys, torch
import torch.nn.functional as F
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_clusters_results.json'
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

    hk = H[0].mlp.register_forward_hook(mlp1_hook)

    sums = torch.zeros(V, D, device=DEV)
    cnts = torch.zeros(V, device=DEV)
    caps = {}
    def cap_hook(mod, args, out):
        caps['out'] = out.detach().float()
        return out
    hk2 = H[0].mlp.register_forward_hook(cap_hook)
    CUR['mode'] = None
    for i in range(0, NFIT, 8):
        idx = FITR[i:i + 8, :-1].to(DEV).contiguous()
        fwd(idx)
        o = caps['out']
        flat_t = idx.reshape(-1)
        sums.index_add_(0, flat_t, o.reshape(-1, D))
        cnts.index_add_(0, flat_t, torch.ones_like(flat_t, dtype=torch.float))
    hk2.remove()
    gmean = sums.sum(0) / cnts.sum()
    tab = torch.where(cnts.unsqueeze(1) > 0, sums / cnts.clamp_min(1).unsqueeze(1),
                      gmean.unsqueeze(0))
    seen = (cnts > 0)
    print(f"tokens covered {int(seen.sum())}", flush=True)

    def kmeans(X, K, w, iters=25, seed=41):
        g = torch.Generator(device='cpu').manual_seed(seed)
        idx0 = torch.randperm(X.shape[0], generator=g)[:K].to(X.device)
        C = X[idx0].clone()
        for _ in range(iters):
            dist = torch.cdist(X, C)
            a = dist.argmin(1)
            for k in range(K):
                mk = a == k
                if mk.any():
                    C[k] = (X[mk] * w[mk].unsqueeze(1)).sum(0) / w[mk].sum()
        return a, C

    Xs = tab[seen]; ws = cnts[seen]
    tok_ids = torch.nonzero(seen).squeeze(1)
    assigns = {}
    KS = (1, 4, 16, 64, 256, 1024)
    ktabs = {}
    for K in KS:
        if K == 1:
            full_assign = torch.zeros(V, dtype=torch.long, device=DEV)
            ktabs[K] = gmean.unsqueeze(0)
        else:
            a, C = kmeans(Xs, K, ws)
            full_assign = torch.zeros(V, dtype=torch.long, device=DEV)
            full_assign[tok_ids] = a
            ktabs[K] = C
        assigns[K] = full_assign
        print(f"K={K} clustered", flush=True)

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

    CUR.update(mean=gmean, tab=tab)
    res = {'full': round(ce_eval(None), 4), 'mean': round(ce_eval('mean'), 4),
           'table50k': round(ce_eval('t1'), 4)}
    for K in KS:
        CUR['tab'] = ktabs[K][assigns[K]]
        res[f'k{K}'] = round(ce_eval('t1'), 4)
        print(f"K={K}: CE {res[f'k{K}']}", flush=True)
    hk.remove()
    stake = res['mean'] - res['full']
    rec = {k: round((res['mean'] - v) / max(stake, 1e-6), 4) for k, v in res.items()
           if k not in ('full', 'mean')}
    import tiktoken; enc = tiktoken.get_encoding('gpt2')
    dig_ids = [t for t in range(V) if seen[t] and enc.decode([t]).strip().isdigit()
               and len(enc.decode([t]).strip()) == 1]
    dig_clusters = set(int(assigns[16][t]) for t in dig_ids)
    pa = stake >= 3.0
    pb = rec['table50k'] >= 0.90
    pc = 0.25 <= rec['k16'] <= 0.55
    out = {'n_fit': NFIT, 'n_eval': NR, 'ce': res, 'stake': round(stake, 4),
           'recovery': rec, 'digit_clusters_at_k16': sorted(dig_clusters),
           'pred_a_big_stake': bool(pa), 'pred_b_table_high': bool(pb),
           'pred_c_loglinear': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stake {stake:.4f} | recovery {rec} | digit clusters@16 {sorted(dig_clusters)}")
    print(f"pred_a stake {pa} | pred_b table {pb} | pred_c loglinear {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
