"""USER-DIRECTED (the convincing version of the §1098 cluster map): are mlp0's class clusters DIFFERENTIALLY READ
downstream? Aggregate class-reading is established (attn1 94% class-reader §784; mlp1 2.9x §850; mlp5 4.9x;
readout 13x §851) — but nothing shows that e.g. the NUMBER code and the PUNCTUATION code are read by different
downstream weights, or that deleting ONE class's code produces CLASS-SPECIFIC damage. Two parts:
(1) WEIGHT READ MAP: for each downstream reader weight W_R (mlp1/mlp5/mlp16 Left+Right; attn1/attn5
    c_q/c_k/c_q2/c_k2/c_v), read strength on each class axis = ||W_R c_k|| for the unit class-centroid direction
    c_k of mlp0's token-mean outputs, vs the mean over random unit directions in mlp0's output span (top-64 PCA)
    -> a readers x classes ratio matrix: WHO reads WHICH class.
(2) CAUSAL DAMAGE MATRIX: for each class k, project the c_k direction OUT of mlp0's output at every position;
    measure dCE grouped by the CURRENT token's class j -> D[k][j]. Class-specific reading predicts DIAGONAL
    dominance (deleting the number code hurts at number positions). Controls: random-in-span directions (3 seeds),
    same conditioning. Also: next-token-CLASS distribution KL at class-k positions under class-k deletion.

REGISTERED PREDICTIONS:
  (0) SANITY: random-direction removals produce small, flat damage rows; class axes near-orthogonal (§915).
  (a) DIAGONAL DAMAGE: for >= 5 of the 8 function-form classes (det/prep/pron/conj/aux/number/punct/Capitalized),
      D[k][k] >= 3x the mean off-diagonal D[k][j!=k] AND >= 3x the random-control damage at class-k positions
      -> each class code is read class-specifically (the clusters are causally distinct variables);
  (b) DIFFERENTIAL READERS: in the weight map, >= 3 (reader, class) pairs show ratio >= 2 with reader-specific
      structure (a reader's top class >= 1.5x its own mean over classes) -> named reader-class pairs
      (e.g. 'attn1 QK reads punctuation; mlp1 Left reads numbers');
  (c) if damage is diffuse (no diagonal dominance anywhere), the class code is written as a package and read
      holistically -- the §1098 clusters are real geometry but not separately-read variables (report plainly)."""
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_class_readers_results.json'
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
    v = ABL['vec'].to(o_.dtype)
    y = o_ - ((o_ - ABL['mu'].to(o_.dtype)) @ v).unsqueeze(-1) * v
    return y


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
    cents = {}
    for kk, cname in enumerate(CLASSES):
        mk = tl == kk
        if int(mk.sum()) >= 5: cents[cname] = F.normalize(tm_c[mk].mean(0), dim=0)
    _, _, Vt = torch.linalg.svd(tm_c, full_matrices=False)
    span = Vt[:64]                                     # 64 x D span of mlp0 token-mean space
    g = torch.Generator(device=DEV).manual_seed(0)
    rand_dirs = F.normalize(torch.randn(50, 64, generator=g, device=DEV) @ span, dim=-1)  # random units in span
    # class-axis orthogonality sanity
    C = torch.stack([cents[c] for c in cents]); gram = (C @ C.T).abs()
    offdiag = float((gram - torch.eye(len(C), device=DEV)).abs().max())

    # (1) weight read map
    readers = {}
    for nm, W in [('mlp1.Left', H[1].mlp.Left.weight), ('mlp1.Right', H[1].mlp.Right.weight),
                  ('mlp5.Left', H[5].mlp.Left.weight), ('mlp5.Right', H[5].mlp.Right.weight),
                  ('mlp16.Left', H[16].mlp.Left.weight),
                  ('attn1.q', H[1].attn.c_q.weight), ('attn1.k', H[1].attn.c_k.weight),
                  ('attn1.q2', H[1].attn.c_q2.weight), ('attn1.k2', H[1].attn.c_k2.weight),
                  ('attn1.v', H[1].attn.c_v.weight),
                  ('attn5.q', H[5].attn.c_q.weight), ('attn5.k', H[5].attn.c_k.weight),
                  ('attn5.v', H[5].attn.c_v.weight)]:
        Wf = W.float()
        null = (rand_dirs @ Wf.T).norm(dim=-1).mean()
        row = {c: round(float((Wf @ cents[c]).norm()/null), 2) for c in cents}
        readers[nm] = row
    print("READ MAP (ratio vs random-in-span):", flush=True)
    for nm, row in readers.items(): print(f"  {nm:>11}: {row}", flush=True)

    # (2) causal damage matrix
    hk = H[0].mlp.register_forward_hook(abl_hook)
    ABL['vec'] = None; ABL['mu'] = mu_out
    base, base_by = ce_by_class(blocks, cls_of)
    Dmat = {}; ctrl_rows = []
    for cname in list(cents):
        ABL['vec'] = cents[cname]
        c, by = ce_by_class(blocks, cls_of)
        Dmat[cname] = [round(float(by[j]-base_by[j]), 4) for j in range(len(CLASSES))]
        ABL['vec'] = None
        print(f"ablate {cname:>13}: overall +{c-base:.4f} | at-own-class +{Dmat[cname][CLASSES.index(cname)]:.4f}", flush=True)
    for s in range(3):
        ABL['vec'] = rand_dirs[s]
        c, by = ce_by_class(blocks, cls_of)
        ctrl_rows.append([round(float(by[j]-base_by[j]), 4) for j in range(len(CLASSES))])
        ABL['vec'] = None
    hk.remove()
    ctrl = torch.tensor(ctrl_rows).mean(0)

    diag_ok = []
    for cname in FUNC8:
        if cname not in Dmat: continue
        kk = CLASSES.index(cname); row = Dmat[cname]
        own = row[kk]; off = (sum(row) - own)/(len(row)-1)
        ok = own >= 3*max(off, 1e-4) and own >= 3*max(float(ctrl[kk]), 1e-4)
        diag_ok.append((cname, round(own, 4), round(off, 4), round(float(ctrl[kk]), 4), bool(ok)))
    n_diag = sum(1 for *_, ok in diag_ok if ok)
    pairs = []
    for nm, row in readers.items():
        vals = list(row.values()); mn = sum(vals)/len(vals)
        for c, v in row.items():
            if v >= 2 and v >= 1.5*mn: pairs.append((nm, c, v))
    out = {'base_ce': round(base, 4), 'class_axis_max_offdiag_cos': round(offdiag, 3),
           'read_map': readers, 'damage_matrix': Dmat, 'control_row': [round(float(v), 4) for v in ctrl],
           'classes': CLASSES, 'diag_test': diag_ok, 'n_diagonal': n_diag,
           'reader_class_pairs': pairs,
           'pred_a_diagonal_damage': bool(n_diag >= 5),
           'pred_b_differential_readers': bool(len(pairs) >= 3),
           'runtime_s': round(time.time()-t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"diag-dominant classes {n_diag}/8: {[d[0] for d in diag_ok if d[-1]]}", flush=True)
    print(f"reader-class pairs: {pairs[:10]}", flush=True)
    print(f"pred_a diagonal {out['pred_a_diagonal_damage']} | pred_b readers {out['pred_b_differential_readers']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
