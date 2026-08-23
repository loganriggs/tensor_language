"""WEIGHT-BASED companion to fig_mlp0_class_map (user-requested): mlp0's class map derived from WEIGHTS ONLY —
no data statistics. For each token t, fold: x = rms_norm(wte[t]); block-0 remix x <- λ₀x+λ₁x (x0=x at block 0);
self-attention at T=1 (rotary at position 0, pattern = self only); mlp0(rms_norm(x + attn_out)). That is the
exact context-free mlp0 response, computed by running block 0 on single-token sequences = pure weight algebra.
Compares with the data map (token-conditional means over 96 FineWeb rows): per-token cosine, class-separation
ratio, and congruence of the two maps (correlation of pairwise-distance matrices)."""
import sys, re, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from palette import INK, SECONDARY, MUTED, GRID, SURFACE

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
NSEQ = 96; SEQ = 256; MIN_OCC = 8
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
COLORS = {'determiner': '#3987e5', 'preposition': '#104281', 'pronoun': '#2e9e8f', 'conjunction': '#7a5fb5',
          'aux/be/neg': '#c2703d', 'number': '#e34948', 'punctuation': '#8c2b2b', 'Capitalized': '#b5892e',
          'subword-piece': '#898781', 'content word': '#c9c7c0'}


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


@torch.no_grad()
def main():
    cl.use_state(PT + 'census_state_diverse.pt')
    blocks = cl.fineweb_rows(NSEQ)[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    # data token means (same as data figure)
    cap = []
    hook = m.transformer.h[0].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)) or None)
    ids = []
    for i in range(0, NSEQ, 8):
        x = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); ids.append(x.reshape(-1))
        xx = F.rms_norm(m.transformer.wte(x), (D,)); x0 = xx; v1 = None
        for blk in m.transformer.h: xx, v1 = blk(xx, v1, x0)
    tok = torch.cat(ids, 0); X = torch.cat(cap, 0); cap.clear()
    xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
    xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    keep = (cn >= MIN_OCC).nonzero().squeeze(1)
    labels, tids = [], []
    for tid in keep.cpu().numpy():
        lab = label_token(int(tid))
        if lab is None: continue
        labels.append(lab); tids.append(int(tid))
    labels = np.array(labels); tids_t = torch.tensor(tids, device=DEV)
    data_means = (xb[tids_t] / cn[tids_t].unsqueeze(1)).cpu().numpy()

    # WEIGHT-BASED: run block 0 on single-token sequences (T=1) with the same mlp0 hook
    idx1 = tids_t.view(-1, 1)                    # B,1
    for i in range(0, idx1.shape[0], 512):
        xx = F.rms_norm(m.transformer.wte(idx1[i:i+512]), (D,)); x0 = xx
        m.transformer.h[0](xx, None, x0)
    hook.remove()
    Wt = torch.cat(cap, 0).cpu().numpy()         # B, D (weight-only mlp0 outputs)

    # agreement
    dc = data_means - data_means.mean(0); wc = Wt - Wt.mean(0)
    cos_tok = np.mean([float(np.dot(dc[i], wc[i]) / (np.linalg.norm(dc[i])*np.linalg.norm(wc[i]) + 1e-9))
                       for i in range(len(tids))])
    def pdistcorr(A, B, n=200):
        rng = np.random.default_rng(0); sel = rng.choice(len(A), min(n, len(A)), replace=False)
        da = np.linalg.norm(A[sel, None] - A[None, sel], axis=-1).ravel()
        db = np.linalg.norm(B[sel, None] - B[None, sel], axis=-1).ravel()
        return float(np.corrcoef(da, db)[0, 1])
    congr = pdistcorr(dc, wc)
    def sep(Xc, lbls):
        cents = {c: Xc[lbls == c].mean(0) for c in CLASSES if (lbls == c).sum() >= 5}
        intra = np.mean([np.linalg.norm(Xc[lbls == c] - cents[c], axis=1).mean() for c in cents])
        cl_ = list(cents)
        inter = np.mean([np.linalg.norm(cents[a]-cents[b]) for ii, a in enumerate(cl_) for b in cl_[ii+1:]])
        return inter/intra
    rng = np.random.default_rng(0)
    sep_w = sep(wc, labels); null_w = np.mean([sep(wc, rng.permutation(labels)) for _ in range(20)])
    print(f"weight-vs-data: mean per-token cosine {cos_tok:.3f} | map congruence (pdist corr) {congr:.3f}")
    print(f"weight-map class separation {sep_w:.3f} vs shuffled {null_w:.3f} ({sep_w/null_w:.2f}x)")

    _, S, Vt2 = np.linalg.svd(wc, full_matrices=False)
    P = wc @ Vt2[:2].T
    fig, ax = plt.subplots(figsize=(11, 8.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for cname in CLASSES:
        msk = labels == cname
        if msk.sum() == 0: continue
        z = 1 if cname in ('content word', 'subword-piece', 'Capitalized') else 3
        a = 0.35 if cname in ('content word', 'subword-piece') else 0.85
        ax.scatter(P[msk, 0], P[msk, 1], s=16, c=COLORS[cname], alpha=a, zorder=z,
                   label=f"{cname} ({int(msk.sum())})", linewidths=0)
    freqs = cn[tids_t].cpu().numpy()
    for cname in CLASSES:
        msk = np.where(labels == cname)[0]
        if len(msk) < 5: continue
        cx, cy = P[msk, 0].mean(), P[msk, 1].mean()
        if cname not in ('content word', 'subword-piece'):
            ax.text(cx, cy, cname.upper(), fontsize=11, fontweight='bold', color=INK, zorder=6, ha='center',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=COLORS[cname], alpha=0.9))
        for j in msk[np.argsort(-freqs[msk])][:7 if cname not in ('content word', 'subword-piece') else 4]:
            ax.annotate(repr(enc.decode([tids[j]]))[1:-1], (P[j, 0], P[j, 1]), fontsize=7, color=SECONDARY,
                        zorder=5, xytext=(3, 3), textcoords='offset points')
    ax.legend(loc='best', fontsize=8, framealpha=0.95)
    ax.tick_params(colors=MUTED); [sp.set_color(GRID) for sp in ax.spines.values()]
    ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    ax.set_title(f"mlp0 class map from WEIGHTS ONLY (embedding + block-0 self-attn folded, T=1)\n"
                 f"per-token cosine w/ data map {cos_tok:.2f} · map congruence {congr:.2f} · "
                 f"separation {sep_w:.2f} vs shuffled {null_w:.2f}", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(PT + 'fig_mlp0_class_map_weights.png', dpi=170, facecolor=SURFACE)
    print("wrote fig_mlp0_class_map_weights.png")


if __name__ == '__main__':
    main()
