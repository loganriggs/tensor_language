# parallel_logit: THE SATURATION DISCRIMINATOR (S1585: serial routing refuted;
# hypothesis = heads and slice are PARALLEL pathways into the same class logit,
# and CE saturation makes their joint CE rise sub-additive). Measure the mean
# CLASS-TARGET LOG-PROBABILITY drop (equivalently mean class-position CE rise
# is what we had; here the mean drop of the target logit itself, pre-softmax)
# at class positions for heads-only / slice-only / joint, is and months, full
# original ensembles (S1580), NR=960.
# If parallel-with-saturation is right, LOGIT drops compose near-additively
# even though CE rises composed at 84%.
# Registered predictions:
#   pred_a is: joint mean class-target logit drop >= .95 x (heads + slice).
#   pred_b months: joint logit drop >= .95 x sum.
#   pred_c CE additivity < logit additivity at both (the saturation
#          signature: the gap direction is consistent).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'parallel_logit_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
ABL = {'L': None, 'topu': None, 'mean_contrib': None}
CONSTS = torch.load(PT + 'opt_ablation_consts_all.pt', map_location='cpu')
HSET = {'set': []}


def mk_head_hook(L):
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
FORM = {'L': None, 'V': None, 'lam': None, 'mean_s': 0.0, 'u': None}


def mk_mlp_hook(L):
    def hook(mod, args, output):
        if ABL['L'] == L:
            z = args[0]
            h = (mod.Left(z).float() * mod.Right(z).float())
            sub = h[:, :, ABL['topu']] @ mod.Down.weight.float()[:, ABL['topu']].T
            return (output.float() - sub + ABL['mean_contrib']).to(output.dtype)
        if FORM['L'] == L:
            z = args[0].float()
            zv = z @ FORM['V']                       # [B,T,r]
            s = (zv * zv) @ FORM['lam']              # [B,T]
            return (output.float()
                    - (s - FORM['mean_s']).unsqueeze(-1) * FORM['u']
                    ).to(output.dtype)
        return None
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
    CLS = {'is': rx(r'^ is$'),
           'months': rx(r'^ (January|February|March|April|May|June|July'
                        r'|August|September|October|November|December)$')}
    SITES = {'is': 17, 'months': 17}
    WU = m.lm_head.weight.float().to(DEV)[:50257]

    # capture z at the two sites over the fit rows (kept on CPU), plus class
    # position masks — enough to compute unit stats, CMU, and mean_s for any form.
    store = {}
    def mk_pre(L):
        def hk(mod, args):
            store.setdefault(L, []).append(args[0].detach())
            return None
        return hk
    site_layers = sorted(set(SITES.values()))
    pre_hooks = [H[L].mlp.register_forward_pre_hook(mk_pre(L))
                 for L in site_layers]
    Z = {L: [] for L in site_layers}
    PM = {cn: [] for cn in CLS}
    for i in range(0, 96, 8):
        store.clear()
        bb = FR[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
        fwd(idx)
        for L in site_layers:
            Z[L].append(store[L][0].float().cpu())
        for cn in CLS:
            pm = CLS[cn].to(DEV)[tg]
            pm[:, :64] = False
            PM[cn].append(pm.cpu())
    for hk in pre_hooks:
        hk.remove()
    print("z captured", flush=True)

    hooks = [H[L].mlp.register_forward_hook(mk_mlp_hook(L)) for L in site_layers]

    def measure(mask_v):
        gs = 0.0; gn = 0; cs = 0.0; cn_ = 0; ls = 0.0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            tgt_logit = lo.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn_ += int(cm.sum())
            ls += float(tgt_logit[cm].sum())
        return gs / max(gn, 1), cs / max(cn_, 1), ls / max(cn_, 1)

    res = {}
    oks = {}
    for cname, L in SITES.items():
        heads = [(int(s.split('.')[0]), int(s.split('.')[1]))
                 for s in r2[cname]['W']['heads']]
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
        wdir = u @ H[L].mlp.Down.weight.float()
        Q = Lw.T @ (wdir[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        order = lam.argsort(descending=True)[:8]   # pos_r8 payload
        Vr = V[:, order].contiguous(); lr = lam[order].contiguous()
        ms = 0.0; n0_ = 0
        for zc in Z[L]:
            zg = zc.to(DEV).reshape(-1, D)
            sv_ = ((zg @ Vr) ** 2) @ lr
            ms += float(sv_.sum()); n0_ += sv_.numel()
        FA = {'L': L, 'V': Vr, 'lam': lr, 'mean_s': ms / n0_, 'u': u}

        def cond(hset, form_on):
            HSET['set'] = hset
            if form_on:
                FORM.update(FA)
            g, c, l = measure(mask_v)
            HSET['set'] = []
            FORM['L'] = None
            return g, c, l

        g0, c0, l0 = cond([], False)
        gh, ch, lh = cond(heads, False)
        gf, cf, lf = cond([], True)
        gj, cj, lj = cond(heads, True)
        ce_add = (cj - c0) / max((ch - c0) + (cf - c0), 1e-9)
        lg_add = (l0 - lj) / max((l0 - lh) + (l0 - lf), 1e-9)
        res[cname] = {'logit_drop_heads': round(l0 - lh, 4),
                      'logit_drop_slice': round(l0 - lf, 4),
                      'logit_drop_joint': round(l0 - lj, 4),
                      'ce_additivity': round(ce_add, 3),
                      'logit_additivity': round(lg_add, 3)}
        print(cname, res[cname], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    pa = res['is']['logit_additivity'] >= 0.95
    pb = res['months']['logit_additivity'] >= 0.95
    pc = all(res[cn]['ce_additivity'] < res[cn]['logit_additivity']
             for cn in SITES)
    out = {'res': res, 'pred_a_is_logit_95': bool(pa),
           'pred_b_months_logit_95': bool(pb),
           'pred_c_saturation_signature': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for hk in hooks + head_hooks:
        hk.remove()
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
