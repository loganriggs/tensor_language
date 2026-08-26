# gate_zspace: COMPUTATION-LEVEL shared-gate test (S1577: output-side gate
# corrections are class-private BY CONSTRUCTION; but the pronouns and the gate
# subspaces share one strong input direction, principal cosine .89). Here the
# ablation is in z-SPACE: mean-substitute the shared principal direction at the
# mlp17 INPUT (pre-hook z' = z - (z.v)v + mean(z.v)v), which changes everything
# mlp17 computes from that direction — payloads, gates, all classes at once.
# Control: the pronouns-gate direction LEAST aligned with the-gate subspace
# (4th principal direction, cos .18) — should help pronouns, not the.
# NR=960.
# Registered predictions:
#   pred_a shared-direction z-ablation lowers pronouns class CE by >= .05.
#   pred_b it ALSO lowers the-class CE by >= .02 (the gate computation is
#          shared even though output corrections are private).
#   pred_c double dissociation: the control direction helps pronouns >= .02
#          while |change to the-class| < .01.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gate_zspace_results.json'
NR = 960
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
ABL = {'L': None, 'topu': None, 'mean_contrib': None}
FORM = {'L': None, 'V': None, 'lam': None, 'mean_s': 0.0, 'u': None}
ZP = {'on': False, 'v': None, 'mu': 0.0}


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
    CLS = {'comma': rx(r'^,$'), 'question': rx(r'^\?$| \?$'),
           'semicolon': rx(r'^;$| ;$'), 'pronouns':
           rx(r'^ (he|she|they|He|She|They)$'),
           'is': rx(r'^ is$'), 'the': rx(r'^ the$'),
           'months': rx(r'^ (January|February|March|April|May|June|July|August'
                        r'|September|October|November|December)$'),
           'close_paren': rx(r'^\)$| \)$')}
    U5 = json.load(open(PT + 'circuit_mlp_units5_results.json'))['res']
    SITES = {cn: U5[cn]['mlp'] for cn in
             ('pronouns', 'is', 'the', 'months')}
    for cn in SITES:
        assert SITES[cn] == 17, f'site drift {cn}'
    UREF = {cn: U5[cn]['K64']['rise_class'] for cn in SITES}
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

    L = 17
    GATES = {}
    for cname in ('pronouns', 'the'):
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
        wdir = u @ H[L].mlp.Down.weight.float()
        Q = Lw.T @ (wdir[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        neg = lam.argsort()[:8]
        GATES[cname] = V[:, neg].contiguous()

    M = GATES['pronouns'].T @ GATES['the']
    U_, sv, W_ = torch.linalg.svd(M)
    v_shared = GATES['pronouns'] @ U_[:, 0]
    v_shared = v_shared / v_shared.norm()
    v_ctrl = GATES['pronouns'] @ U_[:, -1]
    v_ctrl = v_ctrl / v_ctrl.norm()
    print('principal cosines:', [round(float(s), 3) for s in sv], flush=True)

    def mk_zp_hook(LL):
        def hook(mod, args):
            if not ZP['on']:
                return None
            z = args[0]
            zv = (z.float() @ ZP['v'])
            znew = z.float() + (ZP['mu'] - zv).unsqueeze(-1) * ZP['v']
            return (znew.to(z.dtype),) + args[1:]
        return hook
    zp_hook = H[L].mlp.register_forward_pre_hook(mk_zp_hook(L))

    def mu_of(v):
        s_ = 0.0; n_ = 0
        for zc in Z[L]:
            zg = zc.to(DEV).reshape(-1, D)
            s_ += float((zg @ v).sum()); n_ += zg.shape[0]
        return s_ / n_

    base = {cn: measure(CLS[cn]) for cn in SITES}
    res = {'principal_cosines': [round(float(s), 3) for s in sv]}
    for nm, v in (('shared', v_shared), ('control', v_ctrl)):
        ZP.update({'on': True, 'v': v, 'mu': mu_of(v)})
        row = {}
        for cn in SITES:
            g1, c1 = measure(CLS[cn])
            row[cn] = {'d_class': round(c1 - base[cn][1], 4),
                       'd_global': round(g1 - base[cn][0], 4)}
        ZP['on'] = False
        res[nm] = row
        print(nm, {cn: row[cn]['d_class'] for cn in row}, flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    zp_hook.remove()
    for hk in hooks:
        hk.remove()

    pa = res['shared']['pronouns']['d_class'] <= -0.05
    pb = res['shared']['the']['d_class'] <= -0.02
    pc = res['control']['pronouns']['d_class'] <= -0.02 and \
        abs(res['control']['the']['d_class']) < 0.01
    out = {'res': res, 'pred_a_shared_helps_pronouns': bool(pa),
           'pred_b_shared_helps_the': bool(pb),
           'pred_c_double_dissociation': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
