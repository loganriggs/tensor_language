# mlp17_clusters: MODULE LADDER, third entry — the same simplicity curve at the TOP of the
# model. Same instrument as mlp1_clusters/mlp0_clusters (per-token mean output table as the
# ceiling + k-means categories K=1..1024, CE recovery vs the mean-ablation stake on held-out
# rows), pointed at H[17].mlp.
#
# WHY THIS MODULE IS THE RISKY ONE. The ladder's first two entries are FRONT modules, and
# both came back token-resolved (§1323: mlp1's table ceiling 94.4%, log-linear to it). But
# the top MLPs are the one place this program has repeatedly found genuinely CONTEXTUAL
# structure that no token table can express: mlp16/mlp17 gains decode to DOCUMENT REGISTER
# (results/13), and mlp17's functional core is a frequency-CALIBRATION axis plus content
# writers (§650, §694-696) whose value depends on what kind of text you are in. If the
# module ladder is measuring something real about modules rather than about the instrument,
# the table CEILING must fall here.
#
# Registered predictions (the ceiling is the headline; the shape is the control):
#   pred_a CEILING FALLS: mlp17's table50k recovery <= 0.75 (mlp1 was 0.944). A token table
#          cannot buy the top module's contextual value.
#   pred_b SHAPE SURVIVES THE FALL: the curve stays log-linear UP TO its own ceiling —
#          k16/table50k in [0.30, 0.60] (mlp1's ratio was 0.41/0.944 = 0.43).
#   pred_c ORGANIZATION IS NOT SEMANTIC-CLASS HERE: single-digit tokens do NOT co-cluster at
#          K=16 (they spread over >= 5 of the 16 clusters), unlike mlp1 where all digits
#          landed in exactly TWO clusters (§1323 pred_c). Top-module token structure should
#          be frequency/register-organized, not category-organized.
# Reported as a diagnostic, not a bar: per-cluster mean log token-frequency spread at K=16
# against a shuffled-assignment null (the affirmative form of pred_c).
import json, time, sys, torch
import torch.nn.functional as F
from collections import Counter
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl

D = 1152; T = 256; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp17_clusters_results.json'
NFIT = 1920; NR = 480; V = 50257
LI = 17                      # <-- the only structural difference from mlp0_clusters
H = m.transformer.h
CUR = {'toks': None, 'mode': None, 'tab': None, 'mean': None}


def mlp_hook(mod, args, out):
    if CUR['mode'] is None:
        return out
    if CUR['mode'] == 'mean':
        return CUR['mean'].to(out.dtype).expand_as(out)
    return CUR['tab'][CUR['toks']].to(out.dtype)


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
    cl.assert_disjoint(FITR, EVR, label='mlp17_clusters')

    hk = H[LI].mlp.register_forward_hook(mlp_hook)

    sums = torch.zeros(V, D, device=DEV)
    cnts = torch.zeros(V, device=DEV)
    caps = {}
    def cap_hook(mod, args, out):
        caps['out'] = out.detach().float()
        return out
    hk2 = H[LI].mlp.register_forward_hook(cap_hook)
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

    # frequency-organization diagnostic at K=16 (affirmative form of pred_c)
    a16 = assigns[16][tok_ids]
    lf = torch.log(cnts[tok_ids].clamp_min(1))
    def spread(assign):
        mus = [float(lf[assign == k].mean()) for k in range(16) if (assign == k).any()]
        return float(torch.tensor(mus).std()) if len(mus) > 1 else 0.0
    g = torch.Generator(device='cpu').manual_seed(7)
    nulls = [spread(a16[torch.randperm(a16.numel(), generator=g).to(a16.device)])
             for _ in range(5)]
    freq_spread = spread(a16); freq_null = sum(nulls) / len(nulls)

    pa = rec['table50k'] <= 0.75
    pb = 0.30 <= (rec['k16'] / max(rec['table50k'], 1e-6)) <= 0.60
    pc = len(dig_clusters) >= 5
    out = {'layer': LI, 'n_fit': NFIT, 'n_eval': NR, 'ce': res, 'stake': round(stake, 4),
           'recovery': rec, 'k16_over_table': round(rec['k16'] / max(rec['table50k'], 1e-6), 4),
           'digit_clusters_at_k16': sorted(dig_clusters),
           'freq_spread_k16': round(freq_spread, 4), 'freq_spread_null': round(freq_null, 4),
           'pred_a_ceiling_falls': bool(pa), 'pred_b_shape_survives': bool(pb),
           'pred_c_digits_spread': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"stake {stake:.4f} | recovery {rec}")
    print(f"digit clusters@16 {sorted(dig_clusters)} | freq spread {freq_spread:.3f} "
          f"vs null {freq_null:.3f}")
    print(f"pred_a ceiling {pa} | pred_b shape {pb} | pred_c digits {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
