# circuit_mlp_units4: NR=1920 VERIFICATION + K SWEEP of the S1566 finding (the
# first MLP-unit circuit membership: question@mlp11 top-64 units, 104.8x at
# NR=960 — a >50x claim, so the S1523 rule requires NR=1920). Sites FIXED from
# circuit_mlp_units3_results.json (question->mlp11, pronouns->mlp17); unit scores
# recomputed with the same class-conditional recipe on the same 96 fit rows.
# K sweep {16, 64, 256}; generalization check on a SECOND held-out row set
# (skip=15000, 960 rows) at K=64.
# Registered predictions:
#   pred_a question@mlp11 K=64 selectivity >= 50x at NR=1920.
#   pred_b question K=64 class rise on the second held-out set within 35% of the
#          primary-set rise (the unit circuit generalizes across corpora slices).
#   pred_c K=16 keeps >= 50% of the K=64 class rise for question (the membership
#          concentrates further inside the 64).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'circuit_mlp_units4_results.json'
NR = 1920
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
    EVR2 = cl.fineweb_rows(960, skip=15000)[:, :T + 1].contiguous()
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    nl = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if '\n' in ENC.decode([t]):
            nl[t] = True
    CLS = {'newline': nl, 'comma': rx(r'^,$'), 'question': rx(r'^\?$| \?$'),
           'pronouns': rx(r'^ (he|she|they|He|She|They)$')}
    WU = m.lm_head.weight.float().to(DEV)[:50257]

    # unit stats for MLPs 4-17 (one pass)
    store = {}
    def mk_pre(L):
        def hk(mod, args):
            store.setdefault(L, []).append(args[0].detach())
            return None
        return hk
    pre_hooks = [H[L].mlp.register_forward_pre_hook(mk_pre(L))
                 for L in range(4, 18)]
    acc1 = {L: 0 for L in range(4, 18)}; acc2 = {L: 0 for L in range(4, 18)}
    ccond = {cn: {L: 0 for L in range(4, 18)} for cn in CLS}
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
        for L in range(4, 18):
            zz = store[L][0]
            hh_ = (H[L].mlp.Left(zz).float() * H[L].mlp.Right(zz).float()) \
                .reshape(-1, H[L].mlp.Left.weight.shape[0])
            acc1[L] = acc1[L] + hh_.sum(0); acc2[L] = acc2[L] + (hh_ * hh_).sum(0)
            for cn in CLS:
                if int(pm_by[cn].sum()):
                    ccond[cn][L] = ccond[cn][L] + hh_[pm_by[cn]].sum(0)
        n0 += 8 * T
    for hk in pre_hooks:
        hk.remove()
    MU = {L: acc1[L] / n0 for L in range(4, 18)}
    SD = {L: (acc2[L] / n0 - MU[L] ** 2).clamp_min(0).sqrt() for L in range(4, 18)}
    CMU = {cn: {L: ccond[cn][L] / max(ncond[cn], 1) for L in range(4, 18)}
           for cn in CLS}
    hooks = [H[L].mlp.register_forward_hook(mk_mlp_hook(L)) for L in range(4, 18)]
    print("stats done", flush=True)

    def measure(mask_v, rows=None, nr=None):
        gs = 0.0; gn = 0; cs = 0.0; cn = 0
        rows_ = EVR if rows is None else rows
        nr_ = NR if nr is None else nr
        for i in range(0, nr_, 8):
            bb = rows_[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn += int(cm.sum())
        return gs / max(gn, 1), cs / max(cn, 1)

    SITES = {'question': 11, 'pronouns': 17}
    v3 = json.load(open(PT + 'circuit_mlp_units3_results.json'))['res']
    for cn in SITES:
        assert v3[cn]['mlp'] == SITES[cn], f"site drift {cn}"
    res = {}
    for cname, L in SITES.items():
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        wdir = (u @ H[L].mlp.Down.weight.float())
        contrib = (CMU[cname][L] - MU[L]) * wdir
        Wd = H[L].mlp.Down.weight.float()
        g0, c0 = measure(mask_v)
        res[cname] = {'mlp': L}
        for K in (16, 64, 256):
            prom = contrib.argsort(descending=True)[:K]
            ABL.update({'L': L, 'topu': prom,
                        'mean_contrib': MU[L][prom] @ Wd[:, prom].T})
            g1, c1 = measure(mask_v)
            ABL['L'] = None
            sel = (c1 - c0) / max(g1 - g0, 1e-6)
            res[cname][f'K{K}'] = {'rise_class': round(c1 - c0, 4),
                                   'rise_global': round(g1 - g0, 4),
                                   'selectivity': round(sel, 2)}
            print(cname, K, res[cname][f'K{K}'], flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
        # generalization: K=64 on the second held-out set
        prom = contrib.argsort(descending=True)[:64]
        g0b, c0b = measure(mask_v, rows=EVR2, nr=960)
        ABL.update({'L': L, 'topu': prom,
                    'mean_contrib': MU[L][prom] @ Wd[:, prom].T})
        g1b, c1b = measure(mask_v, rows=EVR2, nr=960)
        ABL['L'] = None
        res[cname]['K64_set2'] = {'rise_class': round(c1b - c0b, 4),
                                  'rise_global': round(g1b - g0b, 4)}
        print(cname, 'set2', res[cname]['K64_set2'], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    q = res['question']
    pa = q['K64']['selectivity'] >= 50
    r1 = q['K64']['rise_class']; r2 = q['K64_set2']['rise_class']
    pb = abs(r2 - r1) <= 0.35 * max(r1, 1e-6)
    pc = q['K16']['rise_class'] >= 0.50 * r1
    out = {'res': res, 'pred_a_sel50_1920': bool(pa),
           'pred_b_set2_within_35': bool(pb), 'pred_c_K16_half': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
