# gate_semantics: WHAT IS the-GATE DIRECTION? (S1581: one mlp17 input direction
# whose mean-substitution removes .249 of the-class CE at NR=1920.) Characterize
# the certified gate directions (the@mlp17 and pronouns@mlp17 own top-negative
# eigenvectors) as linear objects:
#   1. VOCAB alignment: rank all 50257 tokens by |unembedding . v| — is the
#      class's own token at the top? (The direction may literally be the
#      residual "the-is-coming" axis.)
#   2. STATE statistics: (z.v) at class target positions vs all positions on
#      the fit rows — effect size in std units.
#   3. READOUT: AUC of (z.v) as a classifier for target==class.
# Registered predictions:
#   pred_a ' the' is in the top-20 of 50257 tokens by |WU @ v_the|.
#   pred_b |mean (z.v | target=the) - mean (z.v)| >= 1.0 std of (z.v).
#   pred_c AUC of z.v for target==the >= .70 (the gate reads a genuine
#          the-prediction state, not a frequency artifact).
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'gate_semantics_results.json'
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
    res = {}
    for cname in ('the', 'pronouns'):
        mask_v = CLS[cname]
        u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
        Lw = H[L].mlp.Left.weight.float(); Rw = H[L].mlp.Right.weight.float()
        wdir = u @ H[L].mlp.Down.weight.float()
        Q = Lw.T @ (wdir[:, None] * Rw)
        S = 0.5 * (Q + Q.T)
        lam, V = torch.linalg.eigh(S)
        v = V[:, int(lam.argmin())].contiguous()

        al = (WU @ v)
        top_pos = al.argsort(descending=True)[:20].tolist()
        top_neg = al.argsort()[:20].tolist()
        rank_abs = int((al.abs() > al.abs()[
            ENC.encode(' the')[0] if cname == 'the'
            else ENC.encode(' he')[0]]).sum())
        names_pos = [ENC.decode([t]) for t in top_pos]
        names_neg = [ENC.decode([t]) for t in top_neg]

        svals = []; labels = []
        for zc, pc in zip(Z[L], PM[cname]):
            zg = zc.to(DEV).reshape(-1, D)
            svals.append(zg @ v)
            labels.append(pc.to(DEV).reshape(-1))
        sv = torch.cat(svals); lb = torch.cat(labels)
        mu_all = float(sv.mean()); sd_all = float(sv.std())
        mu_cls = float(sv[lb].mean()) if int(lb.sum()) else float('nan')
        effect = (mu_cls - mu_all) / max(sd_all, 1e-9)
        order = sv.argsort()
        ranks = torch.empty_like(order, dtype=torch.float)
        ranks[order] = torch.arange(len(sv), dtype=torch.float, device=DEV)
        n1 = int(lb.sum()); n0 = len(sv) - n1
        auc = float((ranks[lb].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))
        auc = max(auc, 1 - auc)

        res[cname] = {'vocab_top_pos': names_pos, 'vocab_top_neg': names_neg,
                      'own_token_abs_rank': rank_abs,
                      'effect_size_std': round(effect, 3),
                      'auc': round(auc, 4), 'n_class_positions': n1}
        print(cname, 'rank', rank_abs, 'effect', round(effect, 2),
              'auc', round(auc, 3), flush=True)
        print(' top+', names_pos[:10], flush=True)
        print(' top-', names_neg[:10], flush=True)
        json.dump({'partial': True, 'res': res}, open(OUT, 'w'), indent=1)
    for hk in hooks:
        hk.remove()

    pa = res['the']['own_token_abs_rank'] < 20
    pb = abs(res['the']['effect_size_std']) >= 1.0
    pc = res['the']['auc'] >= 0.70
    out = {'res': res, 'pred_a_vocab_top20': bool(pa),
           'pred_b_effect_1std': bool(pb), 'pred_c_auc_70': bool(pc),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a {pa} | pred_b {pb} | pred_c {pc}")
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
