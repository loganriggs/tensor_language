# mlp_form_verify: NR=1920 VERIFICATION of the S1570 headline (rank-2 eigen
# slice of the question-mark quadratic form at mlp11: class rise .178 with
# NEGATIVE global rise at NR=960 — a zero-cost removal claim, and implicitly
# >50x, so the S1523 rule applies). Rank-2 and rank-8 forms at NR=1920, plus a
# SECOND held-out row set (skip=15000, 960 rows) for generalization.
# Registered predictions:
#   pred_a rank-2 class rise >= .15 at NR=1920.
#   pred_b rank-2 global rise within +-.001 of zero at NR=1920 (the zero-cost
#          claim survives 4x the rows).
#   pred_c second-set rank-2 class rise within 35% of the primary-set rise.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_form_verify_results.json'
NR = 1920
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
    EVR2 = cl.fineweb_rows(960, skip=15000)[:, :T + 1].contiguous()
    FR = cl.fineweb_rows(96, skip=80)[:, :T + 1].contiguous()
    CLS = {'question': rx(r'^\?$| \?$')}
    SITES = {'question': 11}
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

    res = {}
    for cname, L in SITES.items():
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
        Dw = H[L].mlp.Down.weight.float()
        wdir = u @ Dw
        Q = Lw.T @ (wdir[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        order = lam.abs().argsort(descending=True)
        g0, c0 = measure(mask_v)
        g0b, c0b = measure(mask_v, rows=EVR2, nr=960)
        res[cname] = {'mlp': L, 'clean_class_1920': round(c0, 4)}
        for r in (2, 8):
            idxr = order[:r]
            Vr = V[:, idxr].contiguous(); lr = lam[idxr].contiguous()
            ms = 0.0; n0_ = 0
            for zc in Z[L]:
                zg = zc.to(DEV).reshape(-1, D)
                sv_ = ((zg @ Vr) ** 2) @ lr
                ms += float(sv_.sum()); n0_ += sv_.numel()
            FORM.update({'L': L, 'V': Vr, 'lam': lr,
                         'mean_s': ms / n0_, 'u': u})
            g2, c2 = measure(mask_v)
            g2b, c2b = measure(mask_v, rows=EVR2, nr=960)
            FORM['L'] = None
            res[cname][f'form_r{r}'] = {'rise_class_1920': round(c2 - c0, 4),
                                        'rise_global_1920': round(g2 - g0, 5),
                                        'rise_class_set2': round(c2b - c0b, 4),
                                        'rise_global_set2':
                                        round(g2b - g0b, 5)}
            print(cname, r, res[cname][f'form_r{r}'], flush=True)
            json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    q = res['question']['form_r2']
    pa = q['rise_class_1920'] >= 0.15
    pb = abs(q['rise_global_1920']) <= 0.001
    r1 = q['rise_class_1920']; r2 = q['rise_class_set2']
    pc = abs(r2 - r1) <= 0.35 * max(r1, 1e-6)
    out = {'res': res, 'pred_a_rise_15': bool(pa),
           'pred_b_zero_cost': bool(pb), 'pred_c_set2_35': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
