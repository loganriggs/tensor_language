# unit_reuse_clusters: ARE HIDDEN UNITS REUSED ACROSS CLASSES? (user 2026-08-26:
# "maybe we do clusters of hidden dimensions... we want to say these things mean
# something, and they're reused across multiple things.") Compute the S1566
# class-conditional contribution score for EIGHT classes at every deep MLP
# (4-17), then measure:
#   1. SHARING: per layer, the overlap (intersection count / Jaccard) between
#      classes' top-64 unit sets. Reused units = members of >= 2 top-64 sets at
#      the same layer.
#   2. RELATEDNESS ORDERING: punctuation-family pairs should share more than
#      semantically unrelated pairs.
#   3. CAUSAL REUSE: ablate the largest shared set (the pair with the most
#      common top-64 units at one layer) and check BOTH classes' CE rises —
#      a shared unit cluster that is causally load-bearing for two behaviors.
# NR=960 eval rows, 96 fit rows, mean-substitution ablation (S1566 recipe).
# Registered predictions:
#   pred_a some class pair shares >= 10 of its top-64 units at a common layer.
#   pred_b relatedness ordering holds: max-layer Jaccard(question, close_paren)
#          > max-layer Jaccard(months, comma).
#   pred_c ablating the largest shared set raises class CE >= .01 for BOTH
#          classes of its pair (causal reuse, not just score overlap).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'unit_reuse_clusters_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
ABL = {'L': None, 'topu': None, 'mean_contrib': None}


def mk_mlp_hook(L):
    def hook(mod, args, output):
        if ABL['L'] != L:
            return None
        z = args[0]
        h = (mod.Left(z).float() * mod.Right(z).float())
        sub = h[:, :, ABL['topu']] @ mod.Down.weight.float()[:, ABL['topu']].T
        return (output.float() - sub + ABL['mean_contrib']).to(output.dtype)
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    CLS = {'comma': rx(r'^,$'), 'question': rx(r'^\?$| \?$'),
           'semicolon': rx(r'^;$| ;$'), 'pronouns':
           rx(r'^ (he|she|they|He|She|They)$'),
           'is': rx(r'^ is$'), 'the': rx(r'^ the$'),
           'months': rx(r'^ (January|February|March|April|May|June|July|August'
                        r'|September|October|November|December)$'),
           'close_paren': rx(r'^\)$| \)$')}
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    LAYERS = list(range(4, 18))

    store = {}
    def mk_pre(L):
        def hk(mod, args):
            store.setdefault(L, []).append(args[0].detach())
            return None
        return hk
    pre_hooks = [H[L].mlp.register_forward_pre_hook(mk_pre(L)) for L in LAYERS]
    acc1 = {L: 0 for L in LAYERS}
    ccond = {cn: {L: 0 for L in LAYERS} for cn in CLS}
    ncond = {cn: 0 for cn in CLS}
    n0 = 0
    for i in range(0, 96, 8):
        store.clear()
        bb = FR[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        fwd(idx)
        pm_by = {}
        for cn in CLS:
            pm = CLS[cn].to(DEV)[tg]
            pm[:, :64] = False
            pm_by[cn] = pm.reshape(-1)
            ncond[cn] += int(pm.sum())
        for L in LAYERS:
            zz = store[L][0]
            hh = (H[L].mlp.Left(zz).float() * H[L].mlp.Right(zz).float()) \
                .reshape(-1, H[L].mlp.Left.weight.shape[0])
            acc1[L] = acc1[L] + hh.sum(0)
            for cn in CLS:
                if int(pm_by[cn].sum()):
                    ccond[cn][L] = ccond[cn][L] + hh[pm_by[cn]].sum(0)
        n0 += 8 * T
    for hk in pre_hooks:
        hk.remove()
    MU = {L: acc1[L] / n0 for L in LAYERS}
    CMU = {cn: {L: ccond[cn][L] / max(ncond[cn], 1) for L in LAYERS}
           for cn in CLS}
    print("stats done", flush=True)

    # top-64 sets per (class, layer)
    TOP = {}
    for cn in CLS:
        u = WU[CLS[cn].to(DEV)].mean(0); u = u / u.norm()
        for L in LAYERS:
            wdir = u @ H[L].mlp.Down.weight.float()
            contrib = (CMU[cn][L] - MU[L]) * wdir
            TOP[(cn, L)] = set(contrib.argsort(descending=True)[:64].tolist())

    names = list(CLS)
    pair_max = {}
    best = {'n': -1}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ca, cb = names[a], names[b]
            mx = (0, 0.0, None)
            for L in LAYERS:
                inter = len(TOP[(ca, L)] & TOP[(cb, L)])
                jac = inter / len(TOP[(ca, L)] | TOP[(cb, L)])
                if inter > mx[0]:
                    mx = (inter, jac, L)
            pair_max[f'{ca}|{cb}'] = {'shared': mx[0],
                                      'jaccard': round(mx[1], 3),
                                      'layer': mx[2]}
            if mx[0] > best['n']:
                best = {'n': mx[0], 'pair': (ca, cb), 'layer': mx[2]}
    top_pairs = dict(sorted(pair_max.items(),
                            key=lambda kv: -kv[1]['shared'])[:10])
    print('top shared pairs:', json.dumps(top_pairs, indent=1), flush=True)

    # causal reuse test on the largest shared set
    ca, cb = best['pair']; L = best['layer']
    shared = torch.tensor(sorted(TOP[(ca, L)] & TOP[(cb, L)]),
                          dtype=torch.long, device=DEV)
    hooks = [H[LL].mlp.register_forward_hook(mk_mlp_hook(LL)) for LL in LAYERS]

    def measure(mask_v):
        gs = 0.0; gn = 0; cs = 0.0; cn_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn_ += int(cm.sum())
        return gs / max(gn, 1), cs / max(cn_, 1)

    reuse = {'pair': [ca, cb], 'layer': L, 'n_shared': int(shared.numel())}
    Wd = H[L].mlp.Down.weight.float()
    for cn in (ca, cb):
        g0, c0 = measure(CLS[cn])
        ABL.update({'L': L, 'topu': shared,
                    'mean_contrib': MU[L][shared] @ Wd[:, shared].T})
        g1, c1 = measure(CLS[cn])
        ABL['L'] = None
        reuse[cn] = {'rise_class': round(c1 - c0, 4),
                     'rise_global': round(g1 - g0, 4),
                     'selectivity': round((c1 - c0) / max(g1 - g0, 1e-6), 2)}
        print('reuse', cn, reuse[cn], flush=True)
    for hk in hooks:
        hk.remove()

    pa = best['n'] >= 10
    def maxjac(x, y):
        k = f'{x}|{y}' if f'{x}|{y}' in pair_max else f'{y}|{x}'
        return pair_max[k]['jaccard']
    pb = maxjac('question', 'close_paren') > maxjac('months', 'comma')
    pc = reuse[ca]['rise_class'] >= 0.01 and reuse[cb]['rise_class'] >= 0.01
    out = {'pair_max_overlap': pair_max, 'largest_shared': reuse,
           'pred_a_shared_10': bool(pa), 'pred_b_relatedness_order': bool(pb),
           'pred_c_causal_reuse': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
