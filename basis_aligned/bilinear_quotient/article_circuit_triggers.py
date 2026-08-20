"""ARTICLE CIRCUIT TRIGGERS -- which lexical contexts drive the
article decision? Grounds the circuit to exact token inputs, the last
step of "trace it back to the input embedding exactly".

597 causally established that mlp0 cluster 8's article decision is
CONTEXT-driven: zeroing attn0 (an exact previous-token bigram table)
collapses its firing to correlation 0.24, while zeroing a later
attention leaves it exactly unchanged. So the SIGN of cluster 8's
firing (positive -> a/an, negative -> the) at each article position
is set by the preceding context attn0 carries. This identifies WHICH
preceding contexts -- given that previous-token context is already
causally established as the driver, this census decomposes that
established dependence into its lexical parts.

Method: at every article-target position (next token is a/an/the),
record cluster 8's signed activation (sum of its 101 units' h_j) and
the PREVIOUS token (the token at the current position, which attn0
most carries). Group by previous-token surface class and by specific
previous token; report which contexts push cluster 8 positive (toward
a/an) vs negative (toward the).

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY: mlp0 reclustering reproduces cluster 8 exactly
      -- VOIDS on failure;
  (a) LEXICALLY STRUCTURED, NOT UNIFORM: cluster 8's signed activation
      at article positions varies strongly by previous-token class --
      the spread between the most-positive-driving and most-negative-
      driving class means is at least 1 standard deviation of the
      per-position activation. If it were uniform (article choice not
      lexically triggered), this spread would be near zero;
  (b) THE TRIGGERS (no bar, the finding): report the previous-token
      classes and the specific previous tokens with the highest and
      lowest mean cluster-8 activation -- the lexical contexts that
      push toward a/an vs toward the;
  (c) DIRECTION CHECK: the sign of cluster 8's mean activation by
      previous-token class should PREDICT the actual article outcome
      -- positions whose previous-token class drives cluster 8
      positive should have a higher rate of a/an (vs the) as the
      real next token than positions driven negative. Report the
      a/an-rate for the top-positive vs top-negative classes;
  NULL: shuffling the previous-token labels across positions destroys
      the by-class structure (the spread in (a) collapses to near
      zero) -- the structure is about the real previous token, not an
      artifact of grouping."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'article_circuit_triggers_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20
TOK_A, TOK_AN, TOK_THE, TOK_THE2 = 257, 281, 262, 383

DET = {'a', 'an', 'the', 'this', 'that', 'these', 'those', 'his', 'her',
       'its', 'their', 'our', 'your', 'my', 'some', 'any', 'no', 'each',
       'every', 'another', 'both', 'all'}
PREP = {'of', 'to', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'about', 'over', 'after', 'before', 'between', 'through',
        'during', 'under', 'against', 'without', 'within', 'onto',
        'toward', 'towards', 'upon', 'among', 'across', 'behind', 'near'}
BE = {'is', 'was', 'are', 'were', 'be', 'been', 'being', 'am', "'s", "'re"}


def prev_class(s):
    t = s.strip().lower()
    if not t:
        return 'space'
    if t in BE:
        return 'be_verb'
    if t in DET:
        return 'determiner'
    if t in PREP:
        return 'preposition'
    if t[0].isdigit():
        return 'digit'
    if all(not c.isalnum() for c in t):
        return 'punct'
    if s.strip()[:1].isupper():
        return 'capitalized'
    if s.startswith(' '):
        return 'space_word'
    return 'subword'


@torch.no_grad()
def capture_h(fresh, LJ):
    mlp = m.transformer.h[LJ].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    Dw = mlp.Down.weight.float()
    cap = {}
    hk = mlp.register_forward_pre_hook(
        lambda mo_, a_: cap.__setitem__('X', a_[0]))
    hs = []
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        X = cap['X'].float()
        h = (X @ L.T) * (X @ R.T)
        hs.append(h.reshape(-1, h.shape[-1]).cpu())
    hk.remove()
    return torch.cat(hs, dim=0), Dw.cpu()


def recover_cluster8(fresh):
    H, Dw = capture_h(fresh, 0)
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(H.shape[0], generator=g)[:NSAMP]
    Hs = H[perm]
    imp = (Dw.norm(dim=0) * Hs.std(0))
    order = imp.argsort(descending=True)
    topk = order[:TOPK].numpy()
    O = Hs @ Dw.T
    Oc = O - O.mean(0)
    U, S, Vt = torch.linalg.svd(Oc, full_matrices=False)
    cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
    r = min(int((cum < 0.95).sum().item()) + 1, Vt.shape[0])
    Vr = Vt[:r]
    Dw_topk = Dw[:, topk]
    Dw_proj = Vr @ Dw_topk
    Hk = Hs[:, topk]
    coldw2 = (Dw_proj ** 2).sum(0)
    damage = ((Hk ** 2) * coldw2[None, :]).T.numpy()
    dm = damage - damage.mean(1, keepdims=True)
    dstd = damage.std(1, keepdims=True)
    dstd[dstd < 1e-12] = 1e-12
    corr = np.clip((dm @ dm.T) / (damage.shape[1] * dstd * dstd.T), -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    Z = linkage(squareform((dist + dist.T) / 2, checks=False), method='average')
    labels = fcluster(Z, t=NCLUST, criterion='maxclust')
    sizes = sorted(np.bincount(labels)[1:].tolist(), reverse=True)
    expect = [101, 76, 29, 18, 16, 9, 6, 6, 6, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]
    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    c8 = max(by_cluster, key=lambda k: len(by_cluster[k]))
    return by_cluster[c8], sizes == expect


@torch.no_grad()
def cluster8_signed(fresh, cluster8):
    mlp = m.transformer.h[0].mlp
    L = mlp.Left.weight.float()
    R = mlp.Right.weight.float()
    cap = {}
    hk = mlp.register_forward_pre_hook(
        lambda mo_, a_: cap.__setitem__('X', a_[0]))
    out = []
    for i in range(0, fresh.shape[0], 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
        X = cap['X'].float()
        h = (X @ L.T) * (X @ R.T)
        out.append(h[:, :, cluster8].sum(-1).reshape(-1).cpu())
    hk.remove()
    return torch.cat(out)


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    cluster8, repro = recover_cluster8(fresh)
    print(f'(0) reproduced cluster8 (n={len(cluster8)}): '
          f"{'HELD' if repro else 'FAILED -- VOID'}", flush=True)
    if not repro:
        json.dump({'void': 'reclustering mismatch'}, open(OUT, 'w'), indent=1)
        return

    act = cluster8_signed(fresh, cluster8)          # (N_docs*256,)
    cur = fresh[:, :256].reshape(-1)                # token AT each position
    nxt = fresh[:, 1:257].reshape(-1)               # next token (the target)
    art = ((nxt == TOK_A) | (nxt == TOK_AN) |
           (nxt == TOK_THE) | (nxt == TOK_THE2)).numpy()
    is_indef = ((nxt == TOK_A) | (nxt == TOK_AN)).numpy()

    idxs = np.where(art)[0]
    act_a = act.numpy()[idxs]
    prev_tokens = cur.numpy()[idxs]                 # the current token = context
    prev_cls = np.array([prev_class(cl.d1(int(t))) for t in prev_tokens])
    indef_a = is_indef[idxs]
    print(f'{len(idxs)} article positions', flush=True)

    # (a) by-class means
    classes = sorted(set(prev_cls))
    cls_mean = {c: float(act_a[prev_cls == c].mean())
                for c in classes if (prev_cls == c).sum() >= 5}
    cls_n = {c: int((prev_cls == c).sum()) for c in cls_mean}
    overall_std = float(act_a.std())
    spread = max(cls_mean.values()) - min(cls_mean.values())
    pa = spread >= overall_std
    print(f'(a) by-class spread {spread:.3f} vs per-position std '
          f"{overall_std:.3f}: {'HELD' if pa else 'FAILED'}", flush=True)
    for c in sorted(cls_mean, key=lambda k: cls_mean[k]):
        print(f'    {c:>12}: mean act {cls_mean[c]:+.3f} (n={cls_n[c]})',
              flush=True)

    # (b) specific previous tokens
    tok_mean = {}
    for t in set(prev_tokens.tolist()):
        mask = prev_tokens == t
        if mask.sum() >= 8:
            tok_mean[int(t)] = (float(act_a[mask].mean()), int(mask.sum()))
    top_pos = sorted(tok_mean, key=lambda t: -tok_mean[t][0])[:10]
    top_neg = sorted(tok_mean, key=lambda t: tok_mean[t][0])[:10]
    print('  top a/an-driving previous tokens:', flush=True)
    for t in top_pos:
        print(f'    {cl.d1(t)!r}: {tok_mean[t][0]:+.3f} (n={tok_mean[t][1]})',
              flush=True)
    print('  top the-driving previous tokens:', flush=True)
    for t in top_neg:
        print(f'    {cl.d1(t)!r}: {tok_mean[t][0]:+.3f} (n={tok_mean[t][1]})',
              flush=True)

    # (c) direction check: does class-driven sign predict real outcome?
    pos_classes = [c for c in cls_mean if cls_mean[c] > 0]
    neg_classes = [c for c in cls_mean if cls_mean[c] < 0]
    pos_mask = np.isin(prev_cls, pos_classes)
    neg_mask = np.isin(prev_cls, neg_classes)
    indef_rate_pos = float(indef_a[pos_mask].mean()) if pos_mask.sum() else None
    indef_rate_neg = float(indef_a[neg_mask].mean()) if neg_mask.sum() else None
    pc = (indef_rate_pos is not None and indef_rate_neg is not None and
          indef_rate_pos > indef_rate_neg)
    print(f'(c) a/an-rate: cluster8-positive-driven classes '
          f'{indef_rate_pos:.3f} vs negative-driven {indef_rate_neg:.3f}: '
          f"{'HELD -- sign predicts outcome' if pc else 'FAILED'}", flush=True)

    # NULL: shuffle previous-token labels
    rng = np.random.default_rng(4)
    shuf = rng.permutation(prev_cls)
    scls_mean = {c: float(act_a[shuf == c].mean())
                 for c in classes if (shuf == c).sum() >= 5}
    shuf_spread = max(scls_mean.values()) - min(scls_mean.values())
    null_ok = shuf_spread < 0.3 * spread
    print(f'NULL: shuffled by-class spread {shuf_spread:.3f} vs real '
          f"{spread:.3f}: {'ok' if null_ok else 'CHECK'}", flush=True)

    out = {'repro': bool(repro), 'n_article_pos': len(idxs),
           'class_means': cls_mean, 'class_n': cls_n,
           'by_class_spread': spread, 'per_position_std': overall_std,
           'pred_a': bool(pa),
           'top_aan_tokens': [(cl.d1(t), tok_mean[t][0], tok_mean[t][1])
                              for t in top_pos],
           'top_the_tokens': [(cl.d1(t), tok_mean[t][0], tok_mean[t][1])
                              for t in top_neg],
           'indef_rate_pos_driven': indef_rate_pos,
           'indef_rate_neg_driven': indef_rate_neg, 'pred_c': bool(pc),
           'shuffled_spread': shuf_spread, 'null_ok': bool(null_ok),
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
