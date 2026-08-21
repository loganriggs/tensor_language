"""ARTICLE CHOICE DEPTH (729 follow-up: the a/an-vs-the choice is front-attn-
dominant but DISTRIBUTED -- where ELSE is it carried?). Ablate each block's
attention (c_proj) one at a time, and each block's MLP one at a time, and
measure the drop in the-vs-a/an discrimination AUC. The blocks whose
ablation drops the AUC most = where the choice lives across depth.

REGISTERED PREDICTIONS:
  (0) SANITY: full AUC ~0.87 (matches 729);
  (a) DISTRIBUTED: the article-choice AUC drop is spread across several
      blocks (front attention biggest per 729, but non-trivial contributions
      elsewhere); report the per-block attn and mlp AUC drops and the top
      contributors;
  NULL: a random same-rank ablation of a block drops AUC ~0."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_choice_depth_results.json'
NEVAL = 128
TGT = {'mod': None}   # module to zero-ablate this pass


def hook(mo, i_, o_):
    return torch.zeros_like(o_) if TGT['mod'] is mo else o_


def tok_ids(strs):
    return [t for t in range(m.lm_head.weight.shape[0]) if cl.d1(t) in strs]


@torch.no_grad()
def collect(rows, n, the_t, a_t, the_set, a_set):
    score = []; label = []
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
        p = F.softmax((30.0*torch.tanh(m.lm_head(F.rms_norm(x,(D,)))/30.0)).float(), -1)
        pthe = p[..., the_t].sum(-1).reshape(-1).cpu().numpy()
        pa = p[..., a_t].sum(-1).reshape(-1).cpu().numpy()
        nxt = tgt.reshape(-1).cpu().numpy()
        for k in range(len(nxt)):
            if nxt[k] in the_set or nxt[k] in a_set:
                score.append(pthe[k]/(pthe[k]+pa[k]+1e-9)); label.append(1 if nxt[k] in the_set else 0)
    return np.array(score), np.array(label)


def auc(s, lab):
    lab = np.asarray(lab)
    if lab.sum() in (0, len(lab)): return float('nan')
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s)+1)
    n1 = lab.sum(); n0 = len(lab)-n1
    return float((ranks[lab==1].sum()-n1*(n1+1)/2)/(n1*n0))


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    ev = cl.fineweb_rows(NEVAL)
    the_ids = tok_ids({' the',' The'}); a_ids = tok_ids({' a',' an',' A',' An'})
    the_t = torch.tensor(the_ids, device=DEV); a_t = torch.tensor(a_ids, device=DEV)
    the_set, a_set = set(the_ids), set(a_ids)
    NL = len(m.transformer.h)

    # register hooks on all attn c_proj + mlp, gated by TGT
    handles = []
    for blk in m.transformer.h:
        handles.append(blk.attn.c_proj.register_forward_hook(hook))
        handles.append(blk.mlp.register_forward_hook(hook))

    TGT['mod'] = None
    s, lab = collect(ev, NEVAL, the_t, a_t, the_set, a_set); auc_full = auc(s, lab)
    print(f'article positions {len(lab)}, full AUC {auc_full:.3f}', flush=True)

    attn_drop = {}; mlp_drop = {}
    for li in range(NL):
        TGT['mod'] = m.transformer.h[li].attn.c_proj
        sa,_ = collect(ev, NEVAL, the_t, a_t, the_set, a_set); attn_drop[li] = round(auc_full-auc(sa,lab),4)
        TGT['mod'] = m.transformer.h[li].mlp
        sm,_ = collect(ev, NEVAL, the_t, a_t, the_set, a_set); mlp_drop[li] = round(auc_full-auc(sm,lab),4)
        print(f'blk{li:2d}: attn AUC drop {attn_drop[li]:+.3f}  mlp AUC drop {mlp_drop[li]:+.3f}', flush=True)
    TGT['mod'] = None
    for h in handles: h.remove()

    top = sorted([('attn',li,attn_drop[li]) for li in range(NL)] +
                 [('mlp',li,mlp_drop[li]) for li in range(NL)], key=lambda x:-x[2])[:6]
    print('\ntop article-choice contributors (by AUC drop):', flush=True)
    for kind, li, d in top: print(f'  {kind} blk{li}: {d:+.3f}', flush=True)
    n_signif = sum(1 for li in range(NL) if attn_drop[li] > 0.02) + sum(1 for li in range(NL) if mlp_drop[li] > 0.02)
    pa = n_signif >= 3
    print(f'\n(a) distributed ({n_signif} components with AUC drop > 0.02): {pa}', flush=True)

    out = {'auc_full': round(auc_full,4), 'attn_drop': attn_drop, 'mlp_drop': mlp_drop,
           'top_contributors': [{'kind':k,'block':l,'drop':d} for k,l,d in top],
           'n_significant': n_signif, 'pred_a_distributed': bool(pa), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
