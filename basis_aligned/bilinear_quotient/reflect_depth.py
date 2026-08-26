# reflect_depth: DEPTH PROFILE OF THE SIGNED GENDER READOUT (S1589: reflecting
# the gender axis at the mlp17 input costs .49 CE at pronoun positions). Same
# axis v (from the mlp17 pronoun form), same reflection-about-mean, applied at
# ONE MLP input at a time for layers 13-17 (per-layer axis mean from the fit
# rows). Where does the signed readout live?
# Registered predictions:
#   pred_a mlp17's class-CE effect >= 2x any single earlier layer (the signed
#          readout concentrates at the output-adjacent layer).
#   pred_b each of mlp13/14/15 changes class CE by < .10.
#   pred_c the summed effects of mlp13-16 < the mlp17 effect (no distributed
#          equal-share coding of the readout).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'reflect_depth_results.json'
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
    site_layers = [13, 14, 15, 16, 17]
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

    zp_hooks = [H[LL].mlp.register_forward_pre_hook(mk_zp_hook(LL))
                for LL in site_layers]

    def class_ce():
        cs = 0.0; cn_ = 0
        for i in range(0, NR, 8):
            bb = EVR[i:i + 8].to(DEV)
            idx = bb[:, :-1].contiguous(); tg = bb[:, 1:].contiguous()
            lo = fwd(idx).float()
            ce = F.cross_entropy(lo.reshape(-1, lo.shape[-1]), tg.reshape(-1),
                                 reduction='none').view(tg.shape)
            mk = torch.ones_like(tg, dtype=torch.bool); mk[:, :64] = False
            cm = mask_v.to(DEV)[tg] & mk
            cs += float(ce[cm].sum()); cn_ += int(cm.sum())
        return cs / max(cn_, 1)

    c_base = class_ce()
    res = {'class_ce_base': round(c_base, 4), 'effects': {}}
    for LL in site_layers:
        ms_ = 0.0; n_ = 0
        for zc in Z[LL]:
            zg = zc.to(DEV).reshape(-1, D)
            ms_ += float((zg @ v).sum()); n_ += zg.shape[0]
        ZP.update({'on': True, 'L': LL, 'v': v, 'mu': ms_ / n_})
        c1 = class_ce()
        ZP['on'] = False
        res['effects'][f'mlp{LL}'] = round(c1 - c_base, 4)
        print(f"mlp{LL}: {c1 - c_base:+.4f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in zp_hooks + hooks:
        hk.remove()

    eff = {LL: abs(res['effects'][f'mlp{LL}']) for LL in site_layers}
    pa = eff[17] >= 2 * max(eff[LL] for LL in (13, 14, 15, 16))
    pb = all(eff[LL] < 0.10 for LL in (13, 14, 15))
    pc = sum(eff[LL] for LL in (13, 14, 15, 16)) < eff[17]
    out = {'res': res, 'pred_a_concentrated_at_17': bool(pa),
           'pred_b_early_small': bool(pb), 'pred_c_no_equal_share': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
