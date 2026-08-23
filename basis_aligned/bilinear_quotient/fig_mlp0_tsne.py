"""t-SNE companion (user-requested) to the mlp0 class maps: nonlinear 2D embedding of the same token-mean
mlp0 outputs (data-derived, 96 FineWeb rows) side by side with the WEIGHT-ONLY version. Same labels/colors.
t-SNE on 50-dim PCA-whitened inputs, perplexity 30, cosine metric."""
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
from sklearn.manifold import TSNE

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
def collect():
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
    tok = torch.cat(ids, 0); X = torch.cat(cap, 0); cap.clear()
    xb = torch.zeros(V, D, device=DEV); cn = torch.zeros(V, device=DEV)
    xb.index_add_(0, tok, X); cn.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    keep = (cn >= MIN_OCC).nonzero().squeeze(1)
    labels, tids = [], []
    for tid in keep.cpu().numpy():
        lab = label_token(int(tid))
        if lab is None: continue
        labels.append(lab); tids.append(int(tid))
    tt = torch.tensor(tids, device=DEV)
    data_means = (xb[tt] / cn[tt].unsqueeze(1)).cpu().numpy()
    # weight-only outputs (T=1 through block 0)
    for i in range(0, tt.shape[0], 512):
        xx = F.rms_norm(m.transformer.wte(tt[i:i+512].view(-1, 1)), (D,)); x0 = xx
        m.transformer.h[0](xx, None, x0)
    hook.remove()
    Wt = torch.cat(cap, 0).cpu().numpy()
    return np.array(labels), np.array(tids), data_means, Wt, cn[tt].cpu().numpy()


def tsne2(X):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = (U[:, :50] * 1.0)                        # whitened top-50 PCs
    return TSNE(n_components=2, perplexity=30, metric='cosine', init='pca',
                random_state=0).fit_transform(Z)


def main():
    labels, tids, Dm, Wt, freqs = collect()
    Pd = tsne2(Dm); Pw = tsne2(Wt)
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.4), facecolor=SURFACE)
    for ax, P, ttl in zip(axes, [Pd, Pw], ['DATA token-means (96 FineWeb rows)', 'WEIGHTS ONLY (block-0 fold, T=1)']):
        ax.set_facecolor(SURFACE)
        for cname in CLASSES:
            msk = labels == cname
            if msk.sum() == 0: continue
            z = 1 if cname in ('content word', 'subword-piece', 'Capitalized') else 3
            a = 0.4 if cname in ('content word', 'subword-piece') else 0.9
            ax.scatter(P[msk, 0], P[msk, 1], s=18, c=COLORS[cname], alpha=a, zorder=z,
                       label=f"{cname} ({int(msk.sum())})", linewidths=0)
        for cname in CLASSES:
            msk = np.where(labels == cname)[0]
            if len(msk) < 5 or cname in ('content word', 'subword-piece'): continue
            ax.text(P[msk, 0].mean(), P[msk, 1].mean(), cname.upper(), fontsize=11, fontweight='bold',
                    color=INK, zorder=6, ha='center',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=COLORS[cname], alpha=0.9))
            for j in msk[np.argsort(-freqs[msk])][:6]:
                ax.annotate(repr(enc.decode([int(tids[j])]))[1:-1], (P[j, 0], P[j, 1]), fontsize=7,
                            color=SECONDARY, zorder=5, xytext=(3, 3), textcoords='offset points')
        ax.set_title(f"t-SNE of mlp0 outputs — {ttl}", color=INK, fontsize=12)
        ax.tick_params(colors=MUTED); [sp.set_color(GRID) for sp in ax.spines.values()]
        ax.grid(True, color=GRID, lw=0.6, alpha=0.6)
    axes[0].legend(loc='best', fontsize=8, framealpha=0.95)
    fig.suptitle("mlp0 class clusters, nonlinear view (t-SNE, cosine, perplexity 30) — same classes emerge from data and from weights alone",
                 color=INK, fontsize=12, y=0.99)
    fig.tight_layout()
    fig.savefig(PT + 'fig_mlp0_tsne.png', dpi=170, facecolor=SURFACE)
    print("wrote fig_mlp0_tsne.png")


if __name__ == '__main__':
    main()
