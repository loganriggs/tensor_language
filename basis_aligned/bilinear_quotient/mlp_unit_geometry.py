# mlp_unit_geometry: IS THE UNIT THE RIGHT SPLIT? (user 2026-08-26: "we had the
# third-order tensor... we only need a small part of it. Or a small part of the
# decoder, the left or right encoder.") The bilinear MLP is a third-order tensor
# T_kij = sum_u Down_ku Left_ui Right_uj. For a class direction u_c, the exact
# class-logit contribution of the MLP is the QUADRATIC FORM
#   s(z) = z^T Q z,  Q = Left^T diag(u_c @ Down) Right   (D x D),
# whose symmetric part S has a CANONICAL eigenbasis (units do not: the rank
# decomposition of T is gauge-dependent). Two tests at the certified site
# question@mlp11 (secondary pronouns@mlp17, sites from S1566):
#   1. INTERACTION-SUBSPACE ablation: remove (s_r(z) - mean s_r) * u_c from the
#      MLP output, where s_r keeps only the top-r |eigenvalue| pairs of S.
#      r in {2, 8, 32, full}. If small r reproduces the top-64-unit ablation's
#      class rise, the functional object is a tiny slice of the tensor, not a
#      unit set.
#   2. ENCODER RANK: SVD of Left[top64] and Right[top64] rows — do the 64 units
#      read from a low-dimensional input subspace?
# Baselines in the SAME rows: top-64 unit ablation (S1566 recipe), NR=960.
# Registered predictions:
#   pred_a full-form ablation at question@mlp11: class rise >= .08 with
#          selectivity >= 20x (the class-direction quadratic form IS the
#          circuit component the units were approximating).
#   pred_b rank-32 form ablation achieves >= 70% of the full-form class rise
#          at question (a 32-dim interaction subspace suffices; 32*1152 params
#          vs 64*2*1152 for the unit set).
#   pred_c encoder SVD: rank-16 captures >= .80 of the Frobenius energy of BOTH
#          Left[top64] and Right[top64] at question@mlp11.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_unit_geometry_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
ABL = {'L': None, 'topu': None, 'mean_contrib': None}
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
    CLS = {'question': rx(r'^\?$| \?$'),
           'pronouns': rx(r'^ (he|she|they|He|She|They)$')}
    SITES = {'question': 11, 'pronouns': 17}
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
        gs = 0.0; gn = 0; cs = 0.0; cn = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            gs += float(ce[mk & ~cm].sum()); gn += int((mk & ~cm).sum())
            cs += float(ce[cm].sum()); cn += int(cm.sum())
        return gs / max(gn, 1), cs / max(cn, 1)

    res = {}
    for cname, L in SITES.items():
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
        Dw = H[L].mlp.Down.weight.float()
        wdir = u @ Dw                                   # [n_units]

        # unit stats + class-conditional means from the captured z
        acc1 = 0; acc2 = 0; cacc = 0; n0 = 0; nc = 0
        for zc, pc in zip(Z[L], PM[cname]):
            zg = zc.to(DEV)
            hh = ((zg @ Lw.T) * (zg @ Rw.T)).reshape(-1, Lw.shape[0])
            acc1 = acc1 + hh.sum(0); acc2 = acc2 + (hh * hh).sum(0)
            pf = pc.to(DEV).reshape(-1)
            if int(pf.sum()):
                cacc = cacc + hh[pf].sum(0); nc += int(pf.sum())
            n0 += hh.shape[0]
        MU = acc1 / n0
        CMU = cacc / max(nc, 1)
        contrib = (CMU - MU) * wdir
        top64 = contrib.argsort(descending=True)[:64]
        res[cname] = {'mlp': L, 'n_class_fit': nc}

        # encoder rank of the top-64 unit set
        enc = {}
        for nm, Mx in (('Left', Lw[top64]), ('Right', Rw[top64])):
            sv = torch.linalg.svdvals(Mx)
            tot = float((sv * sv).sum())
            enc[nm] = {f'r{r}': round(float((sv[:r] ** 2).sum()) / tot, 4)
                       for r in (4, 8, 16, 32)}
        res[cname]['encoder_energy'] = enc
        print(cname, 'encoder', enc, flush=True)

        # reference: top-64 unit ablation in these rows
        g0, c0 = measure(mask_v)
        ABL.update({'L': L, 'topu': top64,
                    'mean_contrib': MU[top64] @ Dw[:, top64].T})
        g1, c1 = measure(mask_v)
        ABL['L'] = None
        ref_rise = c1 - c0
        res[cname]['unit64'] = {'rise_class': round(ref_rise, 4),
                                'rise_global': round(g1 - g0, 4),
                                'selectivity':
                                round((c1 - c0) / max(g1 - g0, 1e-6), 2)}
        print(cname, 'unit64', res[cname]['unit64'], flush=True)

        # class-direction quadratic form and its eigenbasis
        Q = Lw.T @ (wdir[:, None] * Rw)                 # [D, D]
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        order = lam.abs().argsort(descending=True)
        spec = lam[order]
        res[cname]['eig_mass'] = {f'r{r}':
                                  round(float(spec[:r].abs().sum()
                                              / spec.abs().sum()), 4)
                                  for r in (2, 8, 32, 128)}
        for r in (2, 8, 32, 'full'):
            idxr = order[:r] if r != 'full' else order
            Vr = V[:, idxr].contiguous(); lr = lam[idxr].contiguous()
            ms = 0.0; n0_ = 0
            for zc in Z[L]:
                zg = zc.to(DEV).reshape(-1, D)
                sv_ = ((zg @ Vr) ** 2) @ lr
                ms += float(sv_.sum()); n0_ += sv_.numel()
            FORM.update({'L': L, 'V': Vr, 'lam': lr,
                         'mean_s': ms / n0_, 'u': u})
            g2, c2 = measure(mask_v)
            FORM['L'] = None
            key = f'form_r{r}'
            res[cname][key] = {'rise_class': round(c2 - c0, 4),
                               'rise_global': round(g2 - g0, 4),
                               'selectivity':
                               round((c2 - c0) / max(g2 - g0, 1e-6), 2),
                               'frac_of_unit64':
                               round((c2 - c0) / max(ref_rise, 1e-6), 3)}
            print(cname, key, res[cname][key], flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    q = res['question']
    pa = q['form_rfull']['rise_class'] >= 0.08 and \
        q['form_rfull']['selectivity'] >= 20
    pb = q['form_r32']['rise_class'] >= \
        0.70 * max(q['form_rfull']['rise_class'], 1e-6)
    pc = q['encoder_energy']['Left']['r16'] >= 0.80 and \
        q['encoder_energy']['Right']['r16'] >= 0.80
    out = {'res': res, 'pred_a_form_works': bool(pa),
           'pred_b_rank32_70pct': bool(pb), 'pred_c_encoder_r16_80': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
