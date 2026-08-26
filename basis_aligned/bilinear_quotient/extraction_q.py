# extraction_q: THE EXTRACTION PROPERTY FOR THE QUESTION CIRCUIT (the third
# circuit property; removal and generalization are certified, extraction is
# untested for the joint circuits). Design: replace ALL 162 heads with their
# optimal constants (attention removed; MLPs intact), then reinstate ONLY the
# 5 certified question heads. Extraction recovery = the fraction of the
# attention-ablation class-CE gap recovered by the 5 live heads:
#   rec = (cA - cB) / (cA - c0),  cA = all heads substituted, cB = all but the
#   circuit substituted, c0 = clean. Global recovery graded the same way.
# Control: 3 draws of 5 RANDOM heads (seeded), same measure. NR=960.
# Registered predictions:
#   pred_a class recovery >= .50 (5 of 162 heads recover half of what all
#          attention does for question-mark prediction).
#   pred_b extraction is selective: class recovery >= 3x global recovery.
#   pred_c circuit heads beat the best of 3 random-5 draws by >= 5x on class
#          recovery.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'extraction_q_results.json'
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
    r2 = json.load(open(PT + 'compression_rank2_results.json'))['res']
    circuit = [(int(s.split('.')[0]), int(s.split('.')[1]))
               for s in r2['question']['W']['heads']]
    ALL = [(L_, h_) for L_ in range(18) for h_ in range(9)]
    mask_v = CLS['question']

    def measure2(hset):
        HSET['set'] = hset
        g, c = measure(mask_v)
        HSET['set'] = []
        return g, c

    g0, c0 = measure2([])
    gA, cA = measure2(ALL)
    gB, cB = measure2([p for p in ALL if p not in circuit])
    rec_c = (cA - cB) / max(cA - c0, 1e-9)
    rec_g = (gA - gB) / max(gA - g0, 1e-9)
    res = {'clean': {'global': round(g0, 4), 'class': round(c0, 4)},
           'all_substituted': {'global': round(gA, 4), 'class': round(cA, 4)},
           'circuit_live': {'global': round(gB, 4), 'class': round(cB, 4)},
           'class_recovery': round(rec_c, 4),
           'global_recovery': round(rec_g, 4)}
    print('recovery: class', round(rec_c, 3), 'global', round(rec_g, 3),
          flush=True)
    json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)

    gen = torch.Generator().manual_seed(11)
    rand_recs = []
    for k in range(3):
        idxs = torch.randperm(162, generator=gen)[:5].tolist()
        rnd = [ALL[j] for j in idxs]
        gR, cR = measure2([p for p in ALL if p not in rnd])
        rr = (cA - cR) / max(cA - c0, 1e-9)
        rand_recs.append(round(rr, 4))
        res[f'random_{k}'] = {'heads': [f'{L_}.{h_}' for L_, h_ in rnd],
                              'class_recovery': round(rr, 4)}
        print('random', k, rnd, round(rr, 3), flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks + head_hooks:
        hk.remove()

    pa = rec_c >= 0.50
    pb = rec_c >= 3 * rec_g
    best_r = max(rand_recs)
    pc = rec_c >= 5 * max(best_r, 1e-9) if best_r > 0 else True
    out = {'res': res, 'random_class_recoveries': rand_recs,
           'pred_a_recovery_50': bool(pa), 'pred_b_selective_3x': bool(pb),
           'pred_c_beats_random_5x': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
