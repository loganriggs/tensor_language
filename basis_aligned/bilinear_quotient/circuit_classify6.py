# circuit_classify6: THE CLASSIFIER (GENERALIZATION) LEG FOR SIX MULTI-TOKEN
# CIRCUITS (months, said, is, the, and, digits — greedy-v2 ensembles from
# S1511/13). Per circuit: per-target-token CE rise under ensemble removal vs the
# weights-only membership score (fraction of the unembedding row inside the
# ensemble's output subspace, rank-64); graded by Spearman + FP/FN rates
# (top-30 damaged / top-30 scored, since some classes have < 50 frequent tokens).
#
# Registered predictions:
#   pred_a median Spearman across the six >= .40.
#   pred_b months (single specialist head) >= .60.
#   pred_c median FN rate <= .60.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_classify6_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
HSET = {'set': []}


def mk_hook(L):
    def hook(mod, args):
        hs = [hh for (LL, hh) in HSET['set'] if LL == L]
        if not hs:
            return None
        x = args[0].clone()
        for hh in hs:
            x[:, :, hh * 128:(hh + 1) * 128] = \
                CONSTS[f'head{L}.{hh}'].to(DEV).float().to(x.dtype)
        return (x,)
    return hook


@torch.no_grad()
def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H:
        x, v1 = blk(x, v1, x0)
    return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


CIRCUITS = {
    'months': {'mask': rx(r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$'),
               'heads': [(14, 7)]},
    'said': {'mask': rx(r'^ (said|says|told|asked|replied)$'),
             'heads': [(11, 3), (9, 1), (11, 5), (12, 2)]},
    'is': {'mask': rx(r'^ is$| was$| are$'), 'heads': [(11, 3), (15, 5)]},
    'the': {'mask': rx(r'^ the$| The$|^The$'),
            'heads': [(7, 3), (10, 8), (11, 7)]},
    'and': {'mask': rx(r'^ and$|^ or$|^ but$'),
            'heads': [(10, 5), (16, 8), (7, 3), (9, 5)]},
    'digits': {'mask': rx(r'^ ?[0-9]+$'),
               'heads': [(7, 3), (6, 5), (12, 6), (11, 5)]},
}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in range(18)]
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    WUn = WU.norm(dim=1).clamp_min(1e-6)

    def per_token():
        tsum = torch.zeros(50257); tn = torch.zeros(50257)
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            tgf = tg.cpu().reshape(-1)
            tsum.index_add_(0, tgf, (ce * mk).cpu().reshape(-1))
            tn.index_add_(0, tgf, mk.cpu().reshape(-1).float())
        return tsum, tn

    HSET['set'] = []
    ts0, tn0 = per_token()
    print("clean measured", flush=True)
    res = {}
    for cname, spec in CIRCUITS.items():
        HSET['set'] = spec['heads']
        ts1, _ = per_token()
        HSET['set'] = []
        okk = (tn0 >= 20) & spec['mask']
        toks = torch.nonzero(okk).flatten()
        if len(toks) < 5:
            res[cname] = {'n_tokens': int(len(toks)), 'skip': True}
            continue
        rv = (ts1[toks] - ts0[toks]) / tn0[toks]
        cols = []
        for (L, hh) in spec['heads']:
            W = H[L].attn.c_proj.weight.float().to(DEV)
            cols.append(W[:, hh * 128:(hh + 1) * 128])
        Eimg = torch.cat(cols, 1)
        q = min(64 + 16, Eimg.shape[1])
        Ue, _, _ = torch.svd_lowrank(Eimg, q=q, niter=4)
        P = Ue[:, :min(64, Eimg.shape[1])]
        sv = ((WU @ P).norm(dim=1) / WUn).cpu()[toks]
        rs = torch.argsort(torch.argsort(sv)).float()
        rr = torch.argsort(torch.argsort(rv)).float()
        n = len(toks)
        rho = 1 - 6 * float(((rs - rr) ** 2).sum()) / max(n * (n * n - 1), 1)
        k = min(30, max(5, n // 3))
        q75 = float(sv.quantile(0.75))
        top_dmg = rv.argsort(descending=True)[:k]
        fn = float((sv[top_dmg] < q75).float().mean())
        med_r = float(rv.median())
        top_sc = sv.argsort(descending=True)[:k]
        fp = float((rv[top_sc] <= med_r).float().mean())
        res[cname] = {'n_tokens': n, 'spearman': round(rho, 3),
                      'fn': round(fn, 3), 'fp': round(fp, 3)}
        print(cname, res[cname], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    import statistics
    done = [cn for cn in res if 'spearman' in res[cn]]
    med_rho = statistics.median([res[cn]['spearman'] for cn in done])
    med_fn = statistics.median([res[cn]['fn'] for cn in done])
    pa = med_rho >= 0.40
    pb = res.get('months', {}).get('spearman', -1) >= 0.60
    pc = med_fn <= 0.60
    out = {'res': res, 'median_spearman': round(med_rho, 3),
           'median_fn': round(med_fn, 3),
           'pred_a_med_rho_40': bool(pa), 'pred_b_months_60': bool(pb),
           'pred_c_med_fn_60': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
