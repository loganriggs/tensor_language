# reflect_units: WHERE DOES THE HE-MASS GO, AND IS THE SIGNED CHANNEL
# UNIT-LOCALIZED? (S1589: reflecting the gender axis at the mlp17 input costs
# .49 CE at pronoun positions; he-mass vanishes rather than becoming she.)
#   1. MASS ACCOUNTING: mean probability over the whole pronoun family and the
#      top-15 token GAINERS at class positions under reflection.
#   2. UNIT LOCALIZATION: each unit's odd coefficient is (l_u.v)(r_u.z_perp)
#      + (r_u.v)(l_u.z_perp); rank units by |l_u.v| + |r_u.v| (the static
#      factor) and re-run the reflection with only the top-64 units' inputs
#      reflected (reflection applied to z seen by those units only, via
#      recomputing their contribution) — does a 64-unit subset carry most of
#      the effect?
#   3. SPECIFICITY: reflection about a random direction with matched mean/var.
# NR=960.
# Registered predictions:
#   pred_a total pronoun-family probability at class positions drops >= 30%
#          under full reflection (mass leaves the family).
#   pred_b the top-64 units by |l_u.v|+|r_u.v| carry >= 50% of the class-CE
#          reflection effect.
#   pred_c random-direction reflection changes class CE by < .05 (the effect
#          is axis-specific).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'reflect_units_results.json'
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

    PRONV = mask_v.to(DEV)

    # partial-reflection hook: reflect z only for a subset of units
    SUB = {'on': False, 'topu': None, 'v': None, 'mu': 0.0}

    def mk_sub_hook(LL):
        def hook(mod, args, output):
            if not SUB['on']:
                return None
            z = args[0].float()
            zv = z @ SUB['v']
            zr = z + 2.0 * (SUB['mu'] - zv).unsqueeze(-1) * SUB['v']
            tu = SUB['topu']
            h_old = (z @ mod.Left.weight.float()[tu].T) \
                * (z @ mod.Right.weight.float()[tu].T)
            h_new = (zr @ mod.Left.weight.float()[tu].T) \
                * (zr @ mod.Right.weight.float()[tu].T)
            return (output.float()
                    + (h_new - h_old) @ mod.Down.weight.float()[:, tu].T
                    ).to(output.dtype)
        return hook
    sub_hook = H[L].mlp.register_forward_hook(mk_sub_hook(L))

    def sweep(mode, v_use=None, mu_use=None, topu=None):
        if mode == 'full':
            ZP.update({'on': True, 'L': L, 'v': v_use, 'mu': mu_use})
        elif mode == 'sub':
            SUB.update({'on': True, 'topu': topu, 'v': v_use, 'mu': mu_use})
        PSUM = []; CE = []; TG = []; PALL = []
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            pr = torch.softmax(lo, dim=-1)
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            fam = pr[:, :, :50257][:, :, PRONV].sum(-1)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = PRONV[tg] & mk
            PSUM.append(fam[cm]); CE.append(ce[cm])
            PALL.append(pr[:, :, :50257][cm].sum(0))
        ZP['on'] = False; SUB['on'] = False
        return (float(torch.cat(PSUM).mean()),
                float(torch.cat(CE).mean()),
                torch.stack(PALL).sum(0))

    fam0, ce_b, pv0 = sweep('none')
    fam1, ce_full, pv1 = sweep('full', v, mu_v)
    gain = (pv1 - pv0).argsort(descending=True)[:15]
    gainers = [(ENC.decode([int(t)]), round(float((pv1 - pv0)[t]), 2))
               for t in gain]
    print('family p:', round(fam0, 4), '->', round(fam1, 4),
          '| class CE:', round(ce_b, 4), '->', round(ce_full, 4), flush=True)
    print('top gainers:', gainers, flush=True)

    # unit localization
    Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
    stat = (Lw @ v).abs() + (Rw @ v).abs()
    topu = stat.argsort(descending=True)[:64]
    _, ce_sub, _ = sweep('sub', v, mu_v, topu)
    frac = (ce_sub - ce_b) / max(ce_full - ce_b, 1e-9)
    print('top-64-unit reflection CE:', round(ce_sub, 4),
          'frac of full:', round(frac, 3), flush=True)

    # random-direction control (matched axis mean)
    g = torch.Generator(device='cpu').manual_seed(7)
    vr = torch.randn(D, generator=g).to(DEV)
    vr = vr / vr.norm()
    msr = 0.0; n0r = 0
    for zc in Z[L]:
        zg = zc.to(DEV).reshape(-1, D)
        msr += float((zg @ vr).sum()); n0r += zg.shape[0]
    _, ce_rand, _ = sweep('full', vr, msr / n0r)
    print('random-dir class CE:', round(ce_rand, 4), flush=True)

    sub_hook.remove()
    zp_hook.remove()
    for hk in hooks:
        hk.remove()

    res = {'family_p_base': round(fam0, 4), 'family_p_reflect': round(fam1, 4),
           'class_ce': {'base': round(ce_b, 4), 'full': round(ce_full, 4),
                        'top64_units': round(ce_sub, 4),
                        'random_dir': round(ce_rand, 4)},
           'top64_frac_of_effect': round(frac, 3), 'top_gainers': gainers}
    pa = (fam0 - fam1) / max(fam0, 1e-9) >= 0.30
    pb = frac >= 0.50
    pc = abs(ce_rand - ce_b) < 0.05
    out = {'res': res, 'pred_a_mass_leaves_family': bool(pa),
           'pred_b_unit_localized': bool(pb),
           'pred_c_axis_specific': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
