# compression_rank3: DOES THE UNIT-TRUNCATION COMPRESSION PRESERVE CIRCUITS?
# The user's direct question (2026-08-26): "keeping the top-k MLP components by
# variance/std... does that end up helping, like, circuits?" S1566 answered the
# membership half (units ARE circuit members for some classes); this answers the
# preservation half: truncate ALL deep MLPs 4-17 to their top-2048 units
# (std(h) x ||Down col|| ranking, the ship's plank recipe, zero-shot) and
# re-measure the 8 weights-only top-5 circuit removals (heads PARSED from
# compression_rank2_results.json) with the SAME rows (NR=960, skip=7000).
# Reference (class_rise, selectivity) values come from that same json.
# Registered predictions:
#   pred_a Spearman(class_rise full, class_rise truncated) >= .8 across the 8
#          circuits (the compression preserves the circuit damage ORDERING).
#   pred_b selectivity within a factor of 2 of the full-model value at >= 5/8.
#   pred_c median truncated-model selectivity >= 2x (circuit structure, not just
#          ordering, survives the compression).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'compression_rank3_results.json'
NR = 960
are = sys.modules[type(m.transformer.h[0].attn).__module__].apply_rotary_emb
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
HSET = {'set': []}
TRUNC = {'on': False, 'topu': {}}


def mk_trunc_hook(L):
    def hook(mod, args, output):
        if not TRUNC['on']:
            return None
        z = args[0]
        tu = TRUNC['topu'][L]
        h = (z @ mod.Left.weight.float()[tu].T) * (z @ mod.Right.weight.float()[tu].T)
        return (h @ mod.Down.weight.float()[:, tu].T
                + mod.Down_bias.float()).to(output.dtype)
    return hook


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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    EVR = cl.fineweb_rows(NR, skip=7000)[:, :T + 1].contiguous()
    AR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    CLS = {'comma': rx(r'^,$'), 'question': rx(r'^\?$| \?$'),
           'semicolon': rx(r'^;$'),
           'pronouns': rx(r'^ (he|she|they|He|She|They)$'),
           'is': rx(r'^ is$| was$| are$'), 'the': rx(r'^ the$| The$|^The$'),
           'months': rx(r'^ (January|February|March|April|May|June|July|August|September|October|November|December)$'),
           'close_paren': rx(r'^\)|^ ?\)$')}
    GREEDY_REF = {  # verified class rises of the registry ensembles (NR>=960)
        'comma': 0.1073, 'question': 1.6353, 'semicolon': 0.4973,
        'pronouns': 0.2162, 'is': 0.067, 'the': 0.0353, 'months': 0.0317,
        'close_paren': 0.6548}
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    UD = {}
    for cn, v in CLS.items():
        u = WU[v.to(DEV)].mean(0)
        UD[cn] = u / u.norm()

    # method W scores
    # unit-ranking stats pass (96 rows, deep_units_a recipe)
    HR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    store = {}
    def mk_pre(L):
        def hk(mod, args):
            store.setdefault(L, []).append(args[0].detach())
            return None
        return hk
    pre_hooks = [H[L].mlp.register_forward_pre_hook(mk_pre(L))
                 for L in range(4, 18)]
    acc1 = {L: 0 for L in range(4, 18)}; acc2 = {L: 0 for L in range(4, 18)}
    n0 = 0
    for i in range(0, 96, 8):
        store.clear()
        fwd(HR[i:i + 8, :-1].to(DEV).contiguous())
        for L in range(4, 18):
            z = store[L][0]
            h = (H[L].mlp.Left(z).float() * H[L].mlp.Right(z).float()) \
                .reshape(-1, H[L].mlp.Left.weight.shape[0])
            acc1[L] = acc1[L] + h.sum(0); acc2[L] = acc2[L] + (h * h).sum(0)
        n0 += 8 * T
    for hk in pre_hooks:
        hk.remove()
    for L in range(4, 18):
        mu = acc1[L] / n0
        hstd = (acc2[L] / n0 - mu * mu).clamp_min(0).sqrt()
        score = hstd * H[L].mlp.Down.weight.float().norm(dim=0)
        TRUNC['topu'][L] = score.argsort(descending=True)[:2048]
    trunc_hooks = [H[L].mlp.register_forward_hook(mk_trunc_hook(L))
                   for L in range(4, 18)]
    print("unit stats + truncation hooks ready", flush=True)

    REF = json.load(open(PT + 'compression_rank2_results.json'))['res']
    ENS = {cn: [(int(s.split('.')[0]), int(s.split('.')[1]))
                for s in REF[cn]['W']['heads']] for cn in CLS}

    def top5(S):
        return [(int(i) // 9, int(i) % 9)
                for i in S.flatten().argsort(descending=True)[:5]]

    hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_hook(L))
             for L in range(18)]

    def rises(hset, mask_v):
        def run():
            gs = 0.0; gn_ = 0; cs = 0.0; cn_ = 0
            for i in range(0, NR, 8):
                bb = EVR[i:i + 8].to(DEV)
                idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
                lo = fwd(idx).float()
                ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]),
                                     tg.reshape(-1),
                                     reduction='none').view(tg.shape)
                mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
                cm = mask_v.to(DEV)[tg] & mk
                gs += float(ce[mk & ~cm].sum()); gn_ += int((mk & ~cm).sum())
                cs += float(ce[cm].sum()); cn_ += int(cm.sum())
            return gs / max(gn_, 1), cs / max(cn_, 1)
        HSET['set'] = []
        g0, c0 = run()
        HSET['set'] = hset
        g1, c1 = run()
        HSET['set'] = []
        return c1 - c0, g1 - g0

    TRUNC['on'] = True
    res = {}
    sel_t = {}; rise_t = {}
    for cn in CLS:
        rc, rg = rises(ENS[cn], CLS[cn])
        s_ = rc / max(rg, 1e-6)
        sel_t[cn] = s_; rise_t[cn] = rc
        res[cn] = {'heads': REF[cn]['W']['heads'],
                   'class_rise_trunc': round(rc, 4),
                   'global_rise_trunc': round(rg, 4),
                   'selectivity_trunc': round(s_, 2),
                   'class_rise_full': REF[cn]['W']['class_rise'],
                   'selectivity_full': REF[cn]['W']['selectivity']}
        print(cn, res[cn]['selectivity_trunc'], 'vs full',
              res[cn]['selectivity_full'], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    TRUNC['on'] = False
    for hk in hooks + trunc_hooks:
        hk.remove()

    import statistics
    names = list(CLS)
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos
        return rk
    fv = [REF[cn]['W']['class_rise'] for cn in names]
    tv = [rise_t[cn] for cn in names]
    ra, rb = rank(fv), rank(tv)
    n = len(names)
    rho = 1 - 6 * sum((ra[i] - rb[i]) ** 2 for i in range(n)) / (n * (n * n - 1))
    within2 = sum(1 for cn in names
                  if 0.5 <= sel_t[cn] / max(REF[cn]['W']['selectivity'], 1e-6) <= 2)
    med_t = statistics.median(sel_t.values())
    pa = rho >= 0.8
    pb = within2 >= 5
    pc = med_t >= 2
    out = {'res': res, 'spearman_class_rise': round(rho, 3),
           'sel_within_2x': within2, 'median_sel_trunc': round(med_t, 2),
           'pred_a_spearman_80': bool(pa), 'pred_b_within2x_5of8': bool(pb),
           'pred_c_med_2x': bool(pc), 'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"spearman {rho:.3f} | within2x {within2}/8 | med {med_t:.2f}")
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
