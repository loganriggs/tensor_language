# serial_channel: MECHANISTIC CONFIRMATION OF THE SERIAL PIPELINE (S1584: is-
# heads 11.3/7.8 route through the mlp17 payload subspace). Capture z at the
# mlp17 input on the fit rows TWICE — baseline and with heads 11.3/7.8 optimal-
# constant substituted — and compare the class-conditional elevation of the is
# payload form s_pos8(z) = sum_i lam_i (z.v_i)^2 over the top-8 positive
# eigenpairs. If the pipeline is serial, substituting the heads collapses the
# class-conditional payload activation.
# Controls: (i) months payload form at months positions under the SAME is-head
# substitution (direction specificity); (ii) non-class (global) s_pos8 shift.
# Registered predictions:
#   pred_a is-head substitution reduces the class-conditional elevation
#          (mean s | is-positions minus mean s) of the is payload form by
#          >= 40%.
#   pred_b months payload elevation at months positions changes by < 15%
#          under the same substitution.
#   pred_c the global (non-class) mean of the is payload form shifts by < 20%
#          of the class-elevation change (the heads feed the CLASS signal, not
#          the subspace's background activity).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'serial_channel_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
ABL = {'L': None, 'topu': None, 'mean_contrib': None}
FORM = {'L': None, 'V': None, 'lam': None, 'mean_s': 0.0, 'u': None}
ZP = {'on': False, 'L': None, 'v': None, 'mu': 0.0}
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
    head_hooks = [H[LL].attn.c_proj.register_forward_pre_hook(mk_head_hook(LL))
                  for LL in range(18)]

    def payload_form(cname):
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
        wdir = u @ H[L].mlp.Down.weight.float()
        Q = Lw.T @ (wdir[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        order = lam.argsort(descending=True)[:8]
        return V[:, order].contiguous(), lam[order].contiguous()

    FORMS = {cn: payload_form(cn) for cn in ('is', 'months')}

    def capture_z():
        zs = []
        store2 = {}
        def hk(mod, args):
            store2.setdefault(L, []).append(args[0].detach())
            return None
        h_ = H[L].mlp.register_forward_pre_hook(hk)
        for i in range(0, 96, 8):
            store2.clear()
            fwd(FR[i:i + 8, :-1].to(DEV).contiguous())
            zs.append(store2[L][0].float().reshape(-1, D))
        h_.remove()
        return torch.cat(zs)

    HSET['set'] = []
    z_base = capture_z()
    HSET['set'] = [(11, 3), (7, 8)]
    z_abl = capture_z()
    HSET['set'] = []

    pm = {cn: torch.cat([p.to(DEV).reshape(-1) for p in PM[cn]])
          for cn in ('is', 'months')}

    res = {}
    for cn in ('is', 'months'):
        Vr, lr = FORMS[cn]
        s_b = ((z_base @ Vr) ** 2) @ lr
        s_a = ((z_abl @ Vr) ** 2) @ lr
        msk = pm[cn]
        elev_b = float(s_b[msk].mean() - s_b.mean())
        elev_a = float(s_a[msk].mean() - s_a.mean())
        res[cn] = {'elev_base': round(elev_b, 4), 'elev_abl': round(elev_a, 4),
                   'elev_drop_frac': round(1 - elev_a / max(elev_b, 1e-9), 3),
                   'global_mean_base': round(float(s_b.mean()), 4),
                   'global_mean_abl': round(float(s_a.mean()), 4),
                   'n_class': int(msk.sum())}
        print(cn, res[cn], flush=True)
    for hk in head_hooks + hooks:
        hk.remove()

    d_elev = abs(res['is']['elev_base'] - res['is']['elev_abl'])
    d_glob = abs(res['is']['global_mean_abl'] - res['is']['global_mean_base'])
    pa = res['is']['elev_drop_frac'] >= 0.40
    pb = abs(res['months']['elev_drop_frac']) < 0.15
    pc = d_glob < 0.20 * max(d_elev, 1e-9)
    out = {'res': res, 'pred_a_is_collapse_40': bool(pa),
           'pred_b_months_specific': bool(pb),
           'pred_c_class_signal_not_background': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
