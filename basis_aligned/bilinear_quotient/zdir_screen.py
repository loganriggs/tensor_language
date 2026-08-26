# zdir_screen: SINGLE z-DIRECTION REMOVALS ACROSS ALL 8 CLASSES (S1579: one
# z-direction mean-substitution doubles the output-side gate correction at the
# @mlp17). For each class at its best MLP (sites parsed from units5): take the
# TOP POSITIVE eigenvector of the class form (payload direction) and the TOP
# NEGATIVE eigenvector (gate direction), mean-substitute each single direction
# at the MLP input, NR=960. Assumption registered: S1579 used the cross-class
# principal direction for the; here each class uses its OWN top eigenvectors.
# Plus NR=1920 verification of the@mlp17 gate z-removal on separate rows.
# Registered predictions:
#   pred_a payload-direction z-removal raises class CE >= .05 at >= 3 of 8.
#   pred_b the@mlp17 own-top-negative-eigvec z-removal at NR=1920 <= -.10.
#   pred_c payload z-removal selectivity (rise_class/rise_global) >= 5x at
#          >= 3 of the classes where the class rise is >= .02.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'zdir_screen_results.json'
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

    def mk_zp_hook(LL):
        def hook(mod, args):
            if not ZP['on'] or ZP['L'] != LL:
                return None
            z = args[0]
            zv = (z.float() @ ZP['v'])
            znew = z.float() + (ZP['mu'] - zv).unsqueeze(-1) * ZP['v']
            return (znew.to(z.dtype),) + args[1:]
        return hook
    zp_hooks = [H[LL].mlp.register_forward_pre_hook(mk_zp_hook(LL))
                for LL in site_layers]
    ZP['L'] = None

    def mu_of(v, LL):
        s_ = 0.0; n_ = 0
        for zc in Z[LL]:
            zg = zc.to(DEV).reshape(-1, D)
            s_ += float((zg @ v).sum()); n_ += zg.shape[0]
        return s_ / n_

    res = {}
    for cname, L in SITES.items():
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
        wdir = u @ H[L].mlp.Down.weight.float()
        Q = Lw.T @ (wdir[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        v_pos = V[:, int(lam.argmax())].contiguous()
        v_neg = V[:, int(lam.argmin())].contiguous()
        g0, c0 = measure(mask_v)
        res[cname] = {'mlp': L}
        for nm, v in (('payload_dir', v_pos), ('gate_dir', v_neg)):
            ZP.update({'on': True, 'L': L, 'v': v, 'mu': mu_of(v, L)})
            g1, c1 = measure(mask_v)
            ZP['on'] = False
            res[cname][nm] = {'rise_class': round(c1 - c0, 4),
                              'rise_global': round(g1 - g0, 4)}
            print(cname, nm, res[cname][nm], flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    # NR=1920 verification: the@mlp17 own gate direction
    L = SITES['the']; mask_v = CLS['the']
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
    wdir = u @ H[L].mlp.Down.weight.float()
    Q = Lw.T @ (wdir[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    v_neg = V[:, int(lam.argmin())].contiguous()
    g0, c0 = measure(mask_v, rows=EVRB, nr=1920)
    ZP.update({'on': True, 'L': L, 'v': v_neg, 'mu': mu_of(v_neg, L)})
    g1, c1 = measure(mask_v, rows=EVRB, nr=1920)
    ZP['on'] = False
    res['the_gate_1920'] = {'rise_class': round(c1 - c0, 4),
                            'rise_global': round(g1 - g0, 4)}
    print('the_gate_1920', res['the_gate_1920'], flush=True)
    for hk in zp_hooks + hooks:
        hk.remove()

    pa = sum(1 for cn in SITES
             if res[cn]['payload_dir']['rise_class'] >= 0.05) >= 3
    pb = res['the_gate_1920']['rise_class'] <= -0.10
    eligible = [cn for cn in SITES
                if res[cn]['payload_dir']['rise_class'] >= 0.02]
    pc = sum(1 for cn in eligible
             if res[cn]['payload_dir']['rise_class']
             / max(res[cn]['payload_dir']['rise_global'], 1e-6) >= 5) >= 3
    out = {'res': res, 'pred_a_payload_3of8': bool(pa),
           'pred_b_the_gate_1920': bool(pb),
           'pred_c_selective_3': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
