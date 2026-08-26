# gate_identity: IS THE mlp17 GATE ONE SHARED MECHANISM? (S1575: pronouns and
# the both carry negative-eigenpart gates at mlp17; S1563 saw late-MLP
# suppression generically.) Build each class's neg-r8 gate subspace, then:
#   1. principal angles between gate subspaces (pronouns vs each other class);
#   2. CROSS-ablation: remove class A's gate (correction along u_A) and measure
#      every class's CE — a shared gate should help non-A classes too.
# Sites fixed at mlp17 (pronouns / is / the / months), NR=960.
# Registered predictions:
#   pred_a mean cos of the top-4 principal angles between the pronouns and the
#          gate subspaces >= .5 (the two strongest gates share geometry).
#   pred_b ablating the pronouns gate lowers THE-class CE by >= .02 (cross-
#          class causal transfer of the gate removal).
#   pred_c ablating the pronouns gate lowers the mean class CE of the other
#          three classes by >= .01 (the gate is broad, not pronoun-private).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gate_identity_results.json'
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
    UD = {}
    for cname in SITES:
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        UD[cname] = u
        Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
        wdir = u @ H[L].mlp.Down.weight.float()
        Q = Lw.T @ (wdir[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        neg = lam.argsort()[:8]
        Vr = V[:, neg].contiguous(); lr = lam[neg].contiguous()
        ms = 0.0; n0_ = 0
        for zc in Z[L]:
            zg = zc.to(DEV).reshape(-1, D)
            sv_ = ((zg @ Vr) ** 2) @ lr
            ms += float(sv_.sum()); n0_ += sv_.numel()
        GATES[cname] = {'V': Vr, 'lam': lr, 'mean_s': ms / n0_, 'u': u}

    # principal angles between gate subspaces
    angles = {}
    for cn in SITES:
        if cn == 'pronouns':
            continue
        M = GATES['pronouns']['V'].T @ GATES[cn]['V']
        sv = torch.linalg.svdvals(M)
        angles[cn] = [round(float(s), 4) for s in sv[:4]]
    print('principal cosines vs pronouns gate:', angles, flush=True)

    # cross-ablation: remove class A's gate, measure every class
    res = {'angles_vs_pronouns': angles}
    base = {cn: measure(CLS[cn]) for cn in SITES}
    for a_ in SITES:
        FORM.update({'L': L, 'V': GATES[a_]['V'], 'lam': GATES[a_]['lam'],
                     'mean_s': GATES[a_]['mean_s'], 'u': GATES[a_]['u']})
        row = {}
        for b_ in SITES:
            g1, c1 = measure(CLS[b_])
            row[b_] = {'d_class': round(c1 - base[b_][1], 4),
                       'd_global': round(g1 - base[b_][0], 4)}
        FORM['L'] = None
        res[f'ablate_{a_}_gate'] = row
        print(a_, {b: row[b]['d_class'] for b in row}, flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    pron = res['ablate_pronouns_gate']
    pa = sum(angles['the'][:4]) / 4 >= 0.5
    pb = pron['the']['d_class'] <= -0.02
    others = [cn for cn in SITES if cn != 'pronouns']
    pc = sum(pron[cn]['d_class'] for cn in others) / 3 <= -0.01
    out = {'res': res, 'pred_a_geometry_50': bool(pa),
           'pred_b_transfers_to_the': bool(pb),
           'pred_c_broad_gate': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
