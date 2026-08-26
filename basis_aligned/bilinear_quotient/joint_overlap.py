# joint_overlap: WHICH HEAD OVERLAPS THE PAYLOAD SLICE? (S1580: is and months
# joints are 84% additive — a ~16% head<->slice overlap at the late site.)
# Per-head leave-one-out: for each head h in the top-5 ensemble, measure the
# slice marginal on (ensemble minus h). If dropping h restores the marginal to
# the slice's solo effect, h is the overlap carrier (its removal effect flows
# THROUGH the mlp17 payload subspace). overlap := slice_solo - marginal.
# NR=960.
# Registered predictions:
#   pred_a for is, ONE head carries >= 50% of the overlap (dropping it recovers
#          >= half the marginal deficit).
#   pred_b that head is 11.3 or 7.8 (the two heads absent from the question/
#          pronouns ensembles that composed cleanly).
#   pred_c months also has a >= 40% single-head overlap carrier.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'joint_overlap_results.json'
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

    head_hooks = [H[LL].attn.c_proj.register_forward_pre_hook(mk_head_hook(LL))
                  for LL in range(18)]
    r2 = json.load(open(PT + 'compression_rank2_results.json'))['res']

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
            g, c = measure(mask_v)
            HSET['set'] = []
            FORM['L'] = None
            return g, c

        g0, c0 = cond([], False)
        gf, cf = cond([], True)
        slice_solo = cf - c0
        gh, ch = cond(heads, False)
        gj, cj = cond(heads, True)
        marg_full = cj - ch
        overlap_full = slice_solo - marg_full
        row = {'heads': r2[cname]['W']['heads'],
               'slice_solo': round(slice_solo, 4),
               'marginal_full': round(marg_full, 4),
               'overlap_full': round(overlap_full, 4), 'loo': {}}
        for k, h in enumerate(heads):
            sub = [x for x in heads if x != h]
            gh_, ch_ = cond(sub, False)
            gj_, cj_ = cond(sub, True)
            marg = cj_ - ch_
            rec = (marg - marg_full) / max(overlap_full, 1e-6)
            row['loo'][r2[cname]['W']['heads'][k]] = {
                'marginal': round(marg, 4),
                'overlap_recovered': round(rec, 3)}
            print(cname, r2[cname]['W']['heads'][k], row['loo'][
                r2[cname]['W']['heads'][k]], flush=True)
        res[cname] = row
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    def best_carrier(cn):
        loo = res[cn]['loo']
        h = max(loo, key=lambda x: loo[x]['overlap_recovered'])
        return h, loo[h]['overlap_recovered']

    h_is, rec_is = best_carrier('is')
    h_mo, rec_mo = best_carrier('months')
    pa = rec_is >= 0.50
    pb = h_is in ('11.3', '7.8')
    pc = rec_mo >= 0.40
    out = {'res': res, 'carrier_is': [h_is, round(rec_is, 3)],
           'carrier_months': [h_mo, round(rec_mo, 3)],
           'pred_a_is_50': bool(pa), 'pred_b_carrier_identity': bool(pb),
           'pred_c_months_40': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for hk in hooks + head_hooks:
        hk.remove()
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")
    return


if __name__ == '__main__':
    main()
