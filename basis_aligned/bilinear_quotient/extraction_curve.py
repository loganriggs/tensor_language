# extraction_curve: RECOVERY vs ENSEMBLE SIZE (S1592: the 5-head question
# circuit recovers only .159 of the attention gap — how fast does recovery grow
# with K?). Heads ranked by the weights-only score (S1564 winner: ||u_class @
# c_proj slice||); extraction = all heads optimal-constant substituted except
# the top-K, K in {5, 10, 20, 40}; MLPs intact; question class; NR=960.
# Registered predictions:
#   pred_a recovery is monotone increasing in K (4 of 4 steps).
#   pred_b K=40 (25% of heads) recovers >= .50 of the class gap.
#   pred_c diminishing returns: per-head marginal recovery for K 5->10 exceeds
#          per-head marginal for K 20->40.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'extraction_curve_results.json'
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

    head_hooks = [H[LL].attn.c_proj.register_forward_pre_hook(mk_head_hook(LL))
                  for LL in range(18)]
    mask_v = CLS['question']
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    score = torch.zeros(18, 9)
    for L_ in range(18):
        Wp = H[L_].attn.c_proj.weight.float()
        for h_ in range(9):
            score[L_, h_] = float((u @ Wp[:, h_ * 128:(h_ + 1) * 128]).norm())
    order = [(int(j) // 9, int(j) % 9)
             for j in score.flatten().argsort(descending=True)]
    ALL = [(L_, h_) for L_ in range(18) for h_ in range(9)]

    def measure2(hset):
        HSET['set'] = hset
        g, c = measure(mask_v)
        HSET['set'] = []
        return g, c

    g0, c0 = measure2([])
    gA, cA = measure2(ALL)
    gap = cA - c0
    res = {'clean_class': round(c0, 4), 'all_sub_class': round(cA, 4),
           'gap': round(gap, 4), 'recovery': {}}
    REC = {}
    for K in (5, 10, 20, 40):
        live = set(order[:K])
        gB, cB = measure2([p for p in ALL if p not in live])
        REC[K] = (cA - cB) / max(gap, 1e-9)
        res['recovery'][f'K{K}'] = round(REC[K], 4)
        res[f'K{K}_heads'] = [f'{L_}.{h_}' for L_, h_ in order[:K]]
        print(f"K={K}: recovery {REC[K]:.3f}", flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks + head_hooks:
        hk.remove()

    ks = [5, 10, 20, 40]
    pa = all(REC[ks[j + 1]] > REC[ks[j]] for j in range(3))
    pb = REC[40] >= 0.50
    pc = (REC[10] - REC[5]) / 5 > (REC[40] - REC[20]) / 20
    out = {'res': res, 'pred_a_monotone': bool(pa), 'pred_b_K40_50': bool(pb),
           'pred_c_diminishing': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
