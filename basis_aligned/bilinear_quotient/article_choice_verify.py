"""ARTICLE CHOICE VERIFY (causal test of the other flagship, FINDINGS item
8 / 636,640). Claim: FRONT ATTENTION carries the a/an-vs-the CHOICE (front
MLP the magnitude; block17 calibrates "the"). Verify CAUSALLY, parallel to
728: at positions where an article follows, does the model's relative
P(the) vs P(a/an) discriminate the ACTUAL article, and does ablating front
attention (or front MLP) COLLAPSE that discrimination?

score = P(the)/(P(the)+P(a)+P(an)); label = 1 if next token is "the" else 0
(among article-next positions). AUC full vs front-attn-ablated vs front-mlp-
ablated vs random-subspace-ablated.

REGISTERED PREDICTIONS:
  (0) SANITY: full-model AUC > 0.6 (context discriminates the vs a/an);
  (a) FRONT ATTENTION carries the CHOICE: ablating front attn (block0+1
      c_proj) drops the AUC substantially, MORE than a random ablation and
      MORE than front-MLP ablation (per 636: attn = choice, mlp = magnitude);
  (b) report AUC full / front-attn / front-mlp / random;
  NULL: random same-rank ablation of the same layers barely changes AUC."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_choice_verify_results.json'
NEVAL = 160
MODE = {'k': None}
RAND = {}


def hook_factory(name):
    def hook(mo, i_, o_):
        if MODE['k'] is None: return o_
        if MODE['k'] == 'zero': return torch.zeros_like(o_)
        of = o_.float(); P = RAND[name]; return (of - (of @ P) @ P.T).to(o_.dtype)
    return hook


def tok_ids(strs):
    V = m.lm_head.weight.shape[0]; out = []
    for t in range(V):
        if cl.d1(t) in strs: out.append(t)
    return out


@torch.no_grad()
def collect(rows, n, the_ids, a_ids):
    allart = the_ids + a_ids
    the_t = torch.tensor(the_ids, device=DEV); a_t = torch.tensor(a_ids, device=DEV)
    art_t = torch.tensor(allart, device=DEV)
    score = []; label = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        lg = 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)
        p = F.softmax(lg.float(), -1)
        pthe = p[..., the_t].sum(-1).reshape(-1).cpu().numpy()
        pa = p[..., a_t].sum(-1).reshape(-1).cpu().numpy()
        nxt = tgt.reshape(-1).cpu().numpy()
        the_set = set(the_ids); a_set = set(a_ids)
        for k in range(len(nxt)):
            if nxt[k] in the_set or nxt[k] in a_set:
                sc = pthe[k] / (pthe[k] + pa[k] + 1e-9)
                score.append(sc); label.append(1 if nxt[k] in the_set else 0)
    return np.array(score), np.array(label)


def auc(score, label):
    label = np.asarray(label)
    if label.sum() == 0 or label.sum() == len(label): return float('nan')
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score)+1)
    n1 = label.sum(); n0 = len(label)-n1
    return float((ranks[label == 1].sum() - n1*(n1+1)/2)/(n1*n0))


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    ev = cl.fineweb_rows(NEVAL)
    the_ids = tok_ids({' the', ' The'}); a_ids = tok_ids({' a', ' an', ' A', ' An'})
    print(f'the ids {len(the_ids)}, a/an ids {len(a_ids)}', flush=True)

    attn = {'b0': m.transformer.h[0].attn.c_proj, 'b1': m.transformer.h[1].attn.c_proj}
    mlp = {'m0': m.transformer.h[0].mlp, 'm1': m.transformer.h[1].mlp}
    g = torch.Generator().manual_seed(0)
    for nm in list(attn)+list(mlp): RAND[nm] = torch.linalg.qr(torch.randn(D,128,generator=g))[0].to(DEV)
    ha = {nm: attn[nm].register_forward_hook(hook_factory(nm)) for nm in attn}
    hm = {nm: mlp[nm].register_forward_hook(hook_factory(nm)) for nm in mlp}

    MODE['k'] = None; s, lab = collect(ev, NEVAL, the_ids, a_ids); auc_full = auc(s, lab)
    # front-attn zero (mlp hooks pass-through by leaving MODE affecting all; separate via distinct hook sets)
    # implement by toggling which hooks are active: remove mlp hooks for attn test
    for h in hm.values(): h.remove()
    MODE['k'] = 'zero'; s1,_ = collect(ev, NEVAL, the_ids, a_ids); auc_attn = auc(s1, lab)
    MODE['k'] = 'rand'; s2,_ = collect(ev, NEVAL, the_ids, a_ids); auc_rand = auc(s2, lab)
    for h in ha.values(): h.remove()
    # front-mlp zero
    hm = {nm: mlp[nm].register_forward_hook(hook_factory(nm)) for nm in mlp}
    MODE['k'] = 'zero'; s3,_ = collect(ev, NEVAL, the_ids, a_ids); auc_mlp = auc(s3, lab)
    MODE['k'] = None
    for h in hm.values(): h.remove()

    print(f'article positions: {len(lab)}, "the": {int(np.sum(lab))}', flush=True)
    print(f'the-vs-a/an AUC  full {auc_full:.3f}  front-attn-abl {auc_attn:.3f}  '
          f'front-mlp-abl {auc_mlp:.3f}  rand-abl {auc_rand:.3f}', flush=True)
    d_attn = auc_full-auc_attn; d_mlp = auc_full-auc_mlp; d_rand = auc_full-auc_rand
    p0 = auc_full > 0.6
    pa = d_attn > 1.5*max(d_rand,1e-6) and d_attn > 0.05
    attn_gt_mlp = d_attn > d_mlp
    print(f'\n(0) discriminates (AUC>0.6): {p0}', flush=True)
    print(f'(a) front-attn drives choice (drop {d_attn:.3f} > 1.5x rand {d_rand:.3f}): {pa}; '
          f'attn>mlp: {attn_gt_mlp}', flush=True)
    out = {'auc_full': round(auc_full,4), 'auc_front_attn_ablated': round(auc_attn,4),
           'auc_front_mlp_ablated': round(auc_mlp,4), 'auc_rand_ablated': round(auc_rand,4),
           'drop_attn': round(d_attn,4), 'drop_mlp': round(d_mlp,4), 'drop_rand': round(d_rand,4),
           'n_articles': int(len(lab)), 'n_the': int(np.sum(lab)), 'attn_gt_mlp': bool(attn_gt_mlp),
           'pred_0': bool(p0), 'pred_a': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
