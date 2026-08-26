# circuit_mlp_units5: THE 8-CLASS MLP-UNIT MEMBERSHIP SCREEN. S1568 certified
# question@mlp11 = 16 units at 183x; this screens ALL EIGHT circuit classes for
# MLP-unit components with the same class-conditional recipe (best layer by
# top-64 positive contribution over MLPs 4-17; ablate top-16 and top-64 with
# mean substitution; NR=960 screen — any >50x selectivity claim goes to NR=1920
# before certification per the S1523 rule).
# Registered predictions:
#   pred_a >= 4 of 8 classes show K=64 class-CE rise >= .05 (unit membership is
#          common, not a question-mark quirk).
#   pred_b >= 2 of 8 (excluding question) reach selectivity >= 10x at K=64.
#   pred_c >= 2 of 8 (excluding question) are CONCENTRATED: K=16 rise >= .6 of
#          K=64 rise (sharp small components exist beyond question).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_mlp_units5_results.json'
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

    res = {}
    for cname in CLS:
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        best = (None, -1, None)
        for L in LAYERS:
            wdir = u @ H[L].mlp.Down.weight.float()
            contrib = (CMU[cname][L] - MU[L]) * wdir
            tot = float(contrib.clamp_min(0).topk(64).values.sum())
            if tot > best[1]:
                best = (L, tot, contrib)
        L, _, contrib = best
        Wd = H[L].mlp.Down.weight.float()
        g0, c0 = measure(mask_v)
        res[cname] = {'mlp': L}
        for K in (16, 64):
            prom = contrib.argsort(descending=True)[:K]
            ABL.update({'L': L, 'topu': prom,
                        'mean_contrib': MU[L][prom] @ Wd[:, prom].T})
            g1, c1 = measure(mask_v)
            ABL['L'] = None
            res[cname][f'K{K}'] = {'rise_class': round(c1 - c0, 4),
                                   'rise_global': round(g1 - g0, 4),
                                   'selectivity':
                                   round((c1 - c0) / max(g1 - g0, 1e-6), 2)}
            print(cname, K, res[cname][f'K{K}'], flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    others = [cn for cn in CLS if cn != 'question']
    pa = sum(1 for cn in CLS if res[cn]['K64']['rise_class'] >= 0.05) >= 4
    pb = sum(1 for cn in others
             if res[cn]['K64']['selectivity'] >= 10) >= 2
    pc = sum(1 for cn in others
             if res[cn]['K16']['rise_class'] >=
             0.6 * max(res[cn]['K64']['rise_class'], 1e-6)
             and res[cn]['K64']['rise_class'] > 0) >= 2
    out = {'res': res, 'pred_a_4of8_rise05': bool(pa),
           'pred_b_2_sel10': bool(pb),
           'pred_c_2_concentrated': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
