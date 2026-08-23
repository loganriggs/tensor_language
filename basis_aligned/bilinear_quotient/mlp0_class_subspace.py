"""RANK-K upgrade of §1103 (registered there): rank-1 class-centroid deletions were near-harmless (class code
is multi-axis §915 + redundantly written §772). Here: ablate each class's SUBSPACE — top-k PCA (k in {2,4,8}) of
the class's member token-mean mlp0 outputs (centered) — from mlp0's output; same class-conditional damage matrix.
NULLS (matched-rank, the §836 lesson): (i) random rank-k inside the class span; (ii) SHUFFLED-MEMBERSHIP class
subspaces (same sizes, random token membership) — the correct null for "is it the CLASS structure specifically".

REGISTERED PREDICTIONS:
  (0) SANITY: damage grows with k; shuffled-membership null << real class subspaces at matched rank.
  (a) DIAGONAL EMERGES WITH RANK: at k=8, >= 5 of the 8 function-form classes show own-class damage >= 3x mean
      off-diagonal AND >= 3x the shuffled-membership null at own-class positions -> the clusters are separately
      READ causal variables (the user's strong story);
  (b) if at k=8 damage becomes large but FLAT across position classes, the class subspaces overlap a shared
      carrier (package reading confirmed at all ranks; report plainly)."""
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_class_subspace_results.json'
NSEQ = 192; SEQ = 256; MIN_OCC = 8
H = m.transformer.h
enc = tiktoken.get_encoding('gpt2')
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'some', 'any', 'each', 'every', 'no', 'all', 'both'}
PREP = {'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'into', 'over', 'under', 'about', 'after',
        'before', 'between', 'through', 'during', 'against', 'without', 'within', 'upon', 'across', 'off', 'up', 'down', 'out'}
PRON = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'his', 'its', 'their',
        'my', 'your', 'our', 'who', 'whom', 'which', 'what', 'himself', 'herself', 'itself', 'themselves'}
CONJ = {'and', 'or', 'but', 'so', 'because', 'if', 'while', 'although', 'though', 'when', 'where', 'as', 'than',
        'whether', 'nor', 'yet', 'since', 'unless'}
AUX = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'am', 'has', 'have', 'had', 'do', 'does', 'did',
       'will', 'would', 'can', 'could', 'should', 'may', 'might', 'must', 'shall', 'not', "n't"}
CLASSES = ['determiner', 'preposition', 'pronoun', 'conjunction', 'aux/be/neg', 'number', 'punctuation',
           'Capitalized', 'subword-piece', 'content word']
FUNC8 = CLASSES[:8]
ABL = {'vec': None}


def label_token(tid):
    raw = enc.decode([tid]); s = raw.strip()
    if s == '': return None
    low = s.lower()
    if re.fullmatch(r"[0-9][0-9,\.]*", s): return 'number'
    if re.fullmatch(r"[^\w\s]+", s): return 'punctuation'
    if low in DET: return 'determiner'
    if low in PREP: return 'preposition'
    if low in PRON: return 'pronoun'
    if low in CONJ: return 'conjunction'
    if low in AUX: return 'aux/be/neg'
    if s[0].isupper(): return 'Capitalized'
    if not raw.startswith(' ') and s.isalpha(): return 'subword-piece'
    if s.isalpha(): return 'content word'
    return None


def fwd(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def abl_hook(mo, i_, o_):
    if ABL['vec'] is None: return None
    U = ABL['vec'].to(o_.dtype)              # D x k orthonormal
    d = o_ - ABL['mu'].to(o_.dtype)
    return o_ - (d @ U) @ U.T


@torch.no_grad()
def ce_by_class(blocks, cls_of):
    """returns overall ce and per-current-class mean ce"""
    tot = 0.0; n = 0
    sums = torch.zeros(len(CLASSES), device=DEV); cnts = torch.zeros(len(CLASSES), device=DEV)
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].reshape(-1)
        lp = F.log_softmax(fwd(idx).float(), -1)
        ce_tok = -lp.reshape(-1, lp.shape[-1])[torch.arange(tgt.shape[0], device=DEV), tgt]
        cc = cls_of[idx.reshape(-1)]
        for kk in range(len(CLASSES)):
            mk = cc == kk
            sums[kk] += ce_tok[mk].sum(); cnts[kk] += mk.sum()
        tot += float(ce_tok.sum()); n += tgt.shape[0]
    return tot/n, (sums/cnts.clamp_min(1)).cpu()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    # vocab class lookup
    cls_of = torch.full((V,), -1, dtype=torch.long)
    for t in range(min(V, 50257)):
        try:
            lab = label_token(t)
        except Exception:
            continue
        if lab is not None: cls_of[t] = CLASSES.index(lab)
    cls_of = cls_of.to(DEV)

    # mlp0 token-mean outputs -> class centroids + span
    cap = []
    hook = H[0].mlp.register_forward_hook(lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)) or None)
    ids = []
    for i in range(0, 96, 8):
        x = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); ids.append(x.reshape(-1)); fwd(x)
    hook.remove()
    tok = torch.cat(ids, 0); X = torch.cat(cap, 0); cap.clear()
    xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
    xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    mu_out = X.mean(0)
    keep = ((cn >= MIN_OCC) & (cls_of >= 0)).nonzero().squeeze(1)
    tm = xb[keep]/cn[keep].unsqueeze(1); tl = cls_of[keep]
    tm_c = tm - tm.mean(0)
    g = torch.Generator(device=DEV).manual_seed(0)

    def class_subspace(members, k):
        """top-k PCA (D x k, orthonormal) of a set of centered token-mean rows"""
        M = tm_c[members]
        _, _, Vt = torch.linalg.svd(M - M.mean(0), full_matrices=False)
        return Vt[:k].T.contiguous()

    KS = [2, 4, 8]
    hk = H[0].mlp.register_forward_hook(abl_hook)
    ABL['vec'] = None; ABL['mu'] = mu_out
    base, base_by = ce_by_class(blocks, cls_of)
    out = {'base_ce': round(base, 4), 'classes': CLASSES, 'by_k': {}}
    for K in KS:
        Dmat = {}; shuf_rows = []
        for kk, cname in enumerate(CLASSES):
            mk = (tl == kk).nonzero().squeeze(1)
            if int(mk.numel()) < K + 2: continue
            ABL['vec'] = class_subspace(mk, K)
            c, by = ce_by_class(blocks, cls_of)
            Dmat[cname] = [round(float(by[j]-base_by[j]), 4) for j in range(len(CLASSES))]
            ABL['vec'] = None
            print(f"k={K} ablate {cname:>13}: overall +{c-base:.4f} | at-own-class +{Dmat[cname][kk]:.4f}", flush=True)
        # shuffled-membership null: same sizes as the FUNC8 classes, random members
        sizes = [int((tl == CLASSES.index(cn)).sum()) for cn in FUNC8]
        for s in range(3):
            perm = torch.randperm(tl.shape[0], generator=torch.Generator().manual_seed(100+s))
            fake = perm[:max(sizes[s % len(sizes)], K+2)].to(DEV)
            ABL['vec'] = class_subspace(fake, K)
            c, by = ce_by_class(blocks, cls_of)
            shuf_rows.append([round(float(by[j]-base_by[j]), 4) for j in range(len(CLASSES))])
            ABL['vec'] = None
        shuf = torch.tensor(shuf_rows).mean(0)
        diag_ok = []
        for cname in FUNC8:
            if cname not in Dmat: continue
            kk = CLASSES.index(cname); row = Dmat[cname]
            own = row[kk]; off = (sum(row) - own)/(len(row)-1)
            ok = own >= 3*max(off, 1e-4) and own >= 3*max(float(shuf[kk]), 1e-4)
            diag_ok.append((cname, round(own, 4), round(off, 4), round(float(shuf[kk]), 4), bool(ok)))
        n_diag = sum(1 for *_, ok in diag_ok if ok)
        out['by_k'][str(K)] = {'damage_matrix': Dmat, 'shuffled_null_row': [round(float(v), 4) for v in shuf],
                               'diag_test': diag_ok, 'n_diagonal': n_diag}
        print(f"k={K}: diag-dominant {n_diag}/8: {[d[0] for d in diag_ok if d[-1]]}", flush=True)
    hk.remove()
    n8 = out['by_k']['8']['n_diagonal']
    out['pred_a_diagonal_emerges'] = bool(n8 >= 5)
    out['pred_b_package_at_all_ranks'] = bool(n8 <= 2)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"pred_a diagonal-emerges {out['pred_a_diagonal_emerges']} | pred_b package {out['pred_b_package_at_all_ranks']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
