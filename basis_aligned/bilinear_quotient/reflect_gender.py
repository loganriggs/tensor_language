# reflect_gender: THE REFLECTION TEST (S1587: quadratic gates are even in
# their axis — the gender gate reads |gender|, not gender). Intervention:
# REFLECT the gender axis at the mlp17 input, z' = z - 2(z.v)v, flipping the
# gender state seen by mlp17 while leaving the residual stream untouched.
# v is the top-negative eigenvector of the pronoun class form, so the class-
# form (even) pathway is EXACTLY invariant under reflection — any behavioral
# change flows through mlp17's non-even pathways (unit cross-terms, other
# output directions). NR=960.
# Registered predictions:
#   pred_a pronoun-class CE changes by < 1/3 of the gate-removal effect
#          (|change| < .033 vs the .099 removal effect — the class-relevant
#          pathway is the even one).
#   pred_b the he/she preference MOVES TOWARD she at class positions:
#          p(she-side) rises AND p(he-side) falls (the non-even pathways read
#          signed gender; corpus he-dominance means reflection pushes most
#          positions toward the she pole).
#   pred_c global CE cost < .01 (reflection is a mild, direction-local
#          intervention).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'reflect_gender_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
ABL = {'L': None, 'topu': None, 'mean_contrib': None}
FORM = {'L': None, 'V': None, 'lam': None, 'mean_s': 0.0, 'u': None}
ZP = {'on': False, 'L': None, 'v': None, 'mu': 0.0}


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
    EVRB = cl.fineweb_rows(1920, skip=7000)[:, :T + 1].contiguous()
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    CLS = {'comma': rx(r'^,$'), 'question': rx(r'^\?$| \?$'),
           'semicolon': rx(r'^;$| ;$'), 'pronouns':
           rx(r'^ (he|she|they|He|She|They)$'),
           'is': rx(r'^ is$'), 'the': rx(r'^ the$'),
           'months': rx(r'^ (January|February|March|April|May|June|July|August'
                        r'|September|October|November|December)$'),
           'close_paren': rx(r'^\)$| \)$')}
    U5 = json.load(open(PT + 'circuit_mlp_units5_results.json'))['res']
    SITES = {cn: U5[cn]['mlp'] for cn in CLS}
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

    L = 17
    mask_v = CLS['pronouns']
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
    wdir = u @ H[L].mlp.Down.weight.float()
    Q = Lw.T @ (wdir[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    v = V[:, int(lam.argmin())].contiguous()
    ms = 0.0; n0_ = 0
    for zc in Z[L]:
        zg = zc.to(DEV).reshape(-1, D)
        ms += float((zg @ v).sum()); n0_ += zg.shape[0]
    mu_v = ms / n0_

    def mk_zp_hook(LL):
        def hook(mod, args):
            if not ZP['on'] or ZP['L'] != LL:
                return None
            z = args[0]
            zv = (z.float() @ ZP['v'])
            # reflect about the axis mean: zv -> 2*mu - zv
            znew = z.float() + 2.0 * (ZP['mu'] - zv).unsqueeze(-1) * ZP['v']
            return (znew.to(z.dtype),) + args[1:]
        return hook
    zp_hook = H[L].mlp.register_forward_pre_hook(mk_zp_hook(L))

    HE = [ENC.encode(' he')[0], ENC.encode(' He')[0]]
    SHE = [ENC.encode(' she')[0], ENC.encode(' She')[0]]
    PRON = torch.tensor([t for t in range(50257) if mask_v[t]], device=DEV)

    def sweep(gate_off):
        ZP.update({'on': gate_off, 'L': L, 'v': v, 'mu': mu_v})
        PH = []; PS = []; CE = []; TOP5 = []; TG = []
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            pr = torch.softmax(lo, dim=-1)
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            top5 = lo.topk(5, dim=-1).indices
            pron_top5 = mask_v.to(DEV)[top5].any(-1)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            PH.append(pr[:, :, HE].sum(-1)[mk]); PS.append(pr[:, :, SHE].sum(-1)[mk])
            CE.append(ce[mk]); TOP5.append(pron_top5[mk]); TG.append(tg[mk])
        ZP['on'] = False
        return (torch.cat(PH), torch.cat(PS), torch.cat(CE),
                torch.cat(TOP5), torch.cat(TG))

    ph0, ps0, ce0, t50, tg0 = sweep(False)
    ph1, ps1, ce1, t51, tg1 = sweep(True)
    is_cls = mask_v.to(DEV)[tg0]
    almost = (~is_cls) & t50
    back = (~is_cls) & ~t50
    res = {'n': {'class': int(is_cls.sum()), 'almost': int(almost.sum()),
                 'background': int(back.sum())}}
    for nm, msk in (('class', is_cls), ('almost', almost),
                    ('background', back)):
        res[nm] = {'p_he_on': round(float(ph0[msk].mean()), 5),
                   'p_he_off': round(float(ph1[msk].mean()), 5),
                   'p_she_on': round(float(ps0[msk].mean()), 5),
                   'p_she_off': round(float(ps1[msk].mean()), 5),
                   'ce_rise': round(float((ce1[msk] - ce0[msk]).mean()), 4)}
        print(nm, res[nm], flush=True)
    zp_hook.remove()
    for hk in hooks:
        hk.remove()

    d_he = res['class']['p_he_off'] - res['class']['p_he_on']
    d_she = res['class']['p_she_off'] - res['class']['p_she_on']
    d_cls_ce = res['class']['ce_rise']
    pa = abs(d_cls_ce) < 0.033
    pb = (d_she > 0) and (d_he < 0)
    glob_ce = (res['almost']['ce_rise'] * res['n']['almost']
               + res['background']['ce_rise'] * res['n']['background']) \
        / max(res['n']['almost'] + res['n']['background'], 1)
    pc = abs(glob_ce) < 0.01
    out = {'res': res, 'd_he_class': round(d_he, 5),
           'd_she_class': round(d_she, 5),
           'nonclass_ce_change': round(glob_ce, 5),
           'pred_a_even_pathway_invariant': bool(pa),
           'pred_b_flips_toward_she': bool(pb),
           'pred_c_mild_global': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
