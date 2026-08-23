"""FIGURE (user-requested): labeled map of mlp0's class clusters. Computes the token-conditional-mean OUTPUT of
mlp0 (the canonical class subspace object, §767-772/§915) for every token with >=8 occurrences in 96 census rows,
labels each token rule-based with the §825/826 class taxonomy, projects to PCA-2D (two panels: PC1-2, PC3-4),
and writes (a) fig_mlp0_class_map.png with class colors + exemplar token labels + class centroids, (b)
mlp0_clusters.md listing every class's member tokens. Also quantifies separation: per-class centroid silhouette
vs shuffled-label null."""
import sys, json, torch, re
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
# categorical colors in the repo's muted-paper register (chrome from palette.py)
COLORS = {'determiner': '#3987e5', 'preposition': '#104281', 'pronoun': '#2e9e8f', 'conjunction': '#7a5fb5',
          'aux/be/neg': '#c2703d', 'number': '#e34948', 'punctuation': '#8c2b2b', 'Capitalized': '#b5892e',
          'subword-piece': '#898781', 'content word': '#c9c7c0'}


def label_token(tid):
    raw = enc.decode([tid])
    s = raw.strip()
    if s == '': return None                     # whitespace/newline tokens: skip
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
    cap = []
    hook = m.transformer.h[0].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)) or None)
    ids = []
    for i in range(0, NSEQ, 8):
        x = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); ids.append(x.reshape(-1))
        xx = F.rms_norm(m.transformer.wte(x), (D,)); x0 = xx; v1 = None
        for blk in m.transformer.h: xx, v1 = blk(xx, v1, x0)
    hook.remove()
    tok = torch.cat(ids, 0); X = torch.cat(cap, 0)
    xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
    xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    keep = (cn >= MIN_OCC).nonzero().squeeze(1)
    means = (xb[keep] / cn[keep].unsqueeze(1)).cpu().numpy()
    freqs = cn[keep].cpu().numpy(); tids = keep.cpu().numpy()

    labels, rows = [], []
    for j, tid in enumerate(tids):
        lab = label_token(int(tid))
        if lab is None: continue
        labels.append(lab); rows.append(j)
    Xm = means[rows]; fr = freqs[rows]; tt = tids[rows]; labels = np.array(labels)
    Xc = Xm - Xm.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:4].T
    print(f"{len(rows)} tokens mapped | var explained PC1-4: {np.round((S[:4]**2)/ (S**2).sum(), 3)}")

    # separation: silhouette-lite = mean(inter-centroid dist) / mean(intra dist) vs shuffled
    def sep(lbls):
        cents = {c: Xc[lbls == c].mean(0) for c in CLASSES if (lbls == c).sum() >= 5}
        intra = np.mean([np.linalg.norm(Xc[lbls == c] - cents[c], axis=1).mean() for c in cents])
        cl_ = list(cents)
        inter = np.mean([np.linalg.norm(cents[a]-cents[b]) for ii, a in enumerate(cl_) for b in cl_[ii+1:]])
        return inter/intra
    rng = np.random.default_rng(0)
    null = np.mean([sep(rng.permutation(labels)) for _ in range(20)])
    real = sep(labels)
    print(f"class separation inter/intra: {real:.3f} vs shuffled-label {null:.3f} ({real/null:.2f}x)")

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.2), facecolor=SURFACE)
    for ax, (i1, i2), ttl in zip(axes, [(0, 1), (2, 3)], ['PC1 x PC2', 'PC3 x PC4']):
        ax.set_facecolor(SURFACE)
        for cname in CLASSES:
            msk = labels == cname
            if msk.sum() == 0: continue
            z = 1 if cname in ('content word', 'subword-piece', 'Capitalized') else 3
            a = 0.35 if cname in ('content word', 'subword-piece') else 0.85
            ax.scatter(P[msk, i1], P[msk, i2], s=14, c=COLORS[cname], alpha=a, zorder=z,
                       label=f"{cname} ({int(msk.sum())})", linewidths=0)
        # exemplar annotations: top-frequency members of the small classes + centroid labels
        for cname in CLASSES:
            msk = np.where(labels == cname)[0]
            if len(msk) < 5: continue
            cx, cy = P[msk, i1].mean(), P[msk, i2].mean()
            if cname not in ('content word', 'subword-piece'):
                ax.text(cx, cy, cname.upper(), fontsize=11, fontweight='bold', color=INK, zorder=6,
                        ha='center', bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=COLORS[cname], alpha=0.9))
            top = msk[np.argsort(-fr[msk])][:8 if cname not in ('content word', 'subword-piece') else 4]
            for j in top:
                ax.annotate(repr(enc.decode([int(tt[j])]))[1:-1], (P[j, i1], P[j, i2]), fontsize=7,
                            color=SECONDARY, zorder=5, xytext=(3, 3), textcoords='offset points')
        ax.set_title(f"mlp0 token-mean outputs — {ttl}", color=INK, fontsize=13)
        ax.tick_params(colors=MUTED); [sp.set_color(GRID) for sp in ax.spines.values()]
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.95)
    fig.suptitle("What the first MLP writes, by token class — each dot is one token's mean mlp0 output "
                 f"(tokens with ≥{MIN_OCC} occurrences; separation {real:.2f} vs shuffled {null:.2f})",
                 color=INK, fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(PT + 'fig_mlp0_class_map.png', dpi=170, facecolor=SURFACE)
    print("wrote fig_mlp0_class_map.png")

    # membership listing
    with open(PT + 'mlp0_clusters.md', 'w') as f:
        f.write("# mlp0 class clusters — membership listing\n\n")
        f.write(f"Token-conditional-mean mlp0 outputs, {len(rows)} tokens (>= {MIN_OCC} occurrences in "
                f"{NSEQ} FineWeb rows). Class separation inter/intra {real:.3f} vs shuffled-label {null:.3f} "
                f"({real/null:.2f}x). PC1-4 variance: {np.round((S[:4]**2)/(S**2).sum(),3).tolist()}.\n\n")
        for cname in CLASSES:
            msk = np.where(labels == cname)[0]
            top = msk[np.argsort(-fr[msk])][:30]
            toks = ", ".join('`' + repr(enc.decode([int(tt[j])]))[1:-1] + '`' for j in top)
            f.write(f"## {cname} — {len(msk)} tokens\n{toks}\n\n")
    print("wrote mlp0_clusters.md")


if __name__ == '__main__':
    main()
