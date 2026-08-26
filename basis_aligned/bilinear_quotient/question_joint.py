# question_joint: DOES THE CERTIFIED MLP SLICE COMPOSE WITH THE ATTENTION
# CIRCUIT? (S1573: question@mlp11 rank-2 slice = .180 rise, zero cost; the
# question attention ensemble = weights-only top-5, S1567.) Measure class/global
# rises for: heads-only (optimal-constant substitution), slice-only, and JOINT
# removal, same rows (NR=960). Heads PARSED from compression_rank2_results.json
# (S1530 rule — no hand transcription).
# Registered predictions:
#   pred_a near-additivity: joint class rise >= .9 x (heads rise + slice rise).
#   pred_b the slice stays free: joint global rise <= heads-only global rise
#          + .002.
#   pred_c non-redundancy: slice marginal on top of heads >= .10 (the MLP
#          member carries class signal the attention ensemble does not).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'question_joint_results.json'
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

    head_hooks = [H[L].attn.c_proj.register_forward_pre_hook(mk_head_hook(L))
                  for L in range(18)]
    r2 = json.load(open(PT + 'compression_rank2_results.json'))['res']
    heads = [(int(s.split('.')[0]), int(s.split('.')[1]))
             for s in r2['question']['W']['heads']]

    L = SITES['question']
    mask_v = CLS['question']
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
    wdir = u @ H[L].mlp.Down.weight.float()
    Q = Lw.T @ (wdir[:, None] * Rw)
    S = 0.5 * (Q + Q.T)
    lam, V = torch.linalg.eigh(S)
    order = lam.abs().argsort(descending=True)[:2]
    Vr = V[:, order].contiguous(); lr = lam[order].contiguous()
    ms = 0.0; n0_ = 0
    for zc in Z[L]:
        zg = zc.to(DEV).reshape(-1, D)
        sv_ = ((zg @ Vr) ** 2) @ lr
        ms += float(sv_.sum()); n0_ += sv_.numel()
    FORM_ARGS = {'L': L, 'V': Vr, 'lam': lr, 'mean_s': ms / n0_, 'u': u}

    def cond(hset, form_on):
        HSET['set'] = hset
        if form_on:
            FORM.update(FORM_ARGS)
        g, c = measure(mask_v)
        HSET['set'] = []
        FORM['L'] = None
        return g, c

    g0, c0 = cond([], False)
    gh, ch = cond(heads, False)
    gf, cf = cond([], True)
    gj, cj = cond(heads, True)
    res = {'heads': r2['question']['W']['heads'],
           'clean': {'global': round(g0, 4), 'class': round(c0, 4)},
           'heads_only': {'rise_class': round(ch - c0, 4),
                          'rise_global': round(gh - g0, 4)},
           'slice_only': {'rise_class': round(cf - c0, 4),
                          'rise_global': round(gf - g0, 4)},
           'joint': {'rise_class': round(cj - c0, 4),
                     'rise_global': round(gj - g0, 4)},
           'slice_marginal_on_heads': round(cj - ch, 4)}
    print(json.dumps(res, indent=1), flush=True)

    pa = (cj - c0) >= 0.9 * ((ch - c0) + (cf - c0))
    pb = (gj - g0) <= (gh - g0) + 0.002
    pc = (cj - ch) >= 0.10
    out = {'res': res, 'pred_a_additive_90': bool(pa),
           'pred_b_slice_free': bool(pb), 'pred_c_marginal_10': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    for hk in hooks + head_hooks:
        hk.remove()
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
