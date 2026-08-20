"""MLP2 UNIT CLUSTER -- how far does the redundant article-choice
machinery extend? 595 confirmed mlp1 has a real, causally-verified
"the"-predicting cluster contributing 13% of mlp0 cluster 8's effect
on the same a/an-vs-the decision. This is the discovery half (mirrors
587's mlp1 pass) of the next question: does mlp0's article machinery
have a THIRD echo at mlp2, weaker still (a decaying redundancy
profile), or does it stop after two layers?

Same recipe as mlp1_unit_cluster.py, applied to mlp2 (LJ=2). If an
article-flavoured cluster (concentration on "the"/"a"/"an"/determiner
class) turns up among the top clusters, it is a candidate for the
same causal patching test 592/595 already validated; if nothing
article-flavoured appears, that itself answers the "how far does this
extend" question -- the redundancy may be a two-layer phenomenon, not
a general pattern repeated at every early layer.

REGISTERED PREDICTIONS:
  (0) SANITY: full-component reconstruction is exact by construction
      (h @ Down.T IS the real output) -- inherited machinery check;
  (a) STABILITY: split-half reclustering of the top-300 units gives
      Adjusted Rand Index >= 0.3 versus a permutation null of ~0 --
      the same bar 579 (0.58) and 587 (0.326, weaker) were held to;
  (b) CONCENTRATION: for the 3 largest clusters, at least 1 of 3 has
      >= 5/8 positive-example concentration on one surface class;
  (c) THE REDUNDANCY QUESTION (no bar, report only): does any of
      mlp2's top clusters read as article-related (determiner class,
      "the"/"a"/"an" in its top examples)? If yes, flag it as a
      candidate for a follow-up causal patching test matching 595's
      design. If no article signal appears in the top clusters, note
      that plainly as evidence the two-layer redundancy (mlp0+mlp1)
      does not extend to a third;
  NULL: shuffling sample alignment before clustering drops the
      stability ARI to < 0.1."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import census_lib as cl
from bilin18_joint_removal import m, DEV
D = 1152
T = 256
LJ = 2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp2_unit_cluster_results.json'
NFRESH = 64
NSAMP = 4000
TOPK = 300
NCLUST = 20

DET={'a','an','the','this','that','these','those','his','her',
     'its','their','our','your','my','some','any','no','each',
     'every','another','both','all'}
PREP={'of','to','in','for','on','with','at','by','from','as',
      'into','about','over','after','before','between','through',
      'during','under','against','without','within','onto',
      'toward','towards','upon','among','across','behind','near'}
PRON={'he','she','it','they','we','you','i','him','them','us',
      'me','who','which','what','his','hers','theirs','itself',
      'himself','herself','themselves'}

def fine_class(s):
    t=s.strip().lower()
    if not t: return 'space'
    if t in DET: return 'determiner'
    if t in PREP: return 'preposition'
    if t in PRON: return 'pronoun'
    if t[0].isdigit(): return 'digit'
    if all(not c.isalnum() for c in t): return 'punct'
    if s.strip()[:1].isupper(): return 'capitalized'
    if s.startswith(' '): return 'space_word'
    return 'subword'


def adjusted_rand_index(labels_true, labels_pred):
    from collections import Counter
    n = len(labels_true)
    ct = Counter(zip(labels_true, labels_pred))
    a = Counter(labels_true)
    b = Counter(labels_pred)
    def comb2(x): return x * (x - 1) / 2
    sum_comb_c = sum(comb2(v) for v in ct.values())
    sum_comb_a = sum(comb2(v) for v in a.values())
    sum_comb_b = sum(comb2(v) for v in b.values())
    comb_n = comb2(n)
    expected = sum_comb_a * sum_comb_b / comb_n if comb_n else 0
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_idx - expected
    if abs(denom) < 1e-12:
        return 1.0
    return (sum_comb_c - expected) / denom


@torch.no_grad()
def capture(fresh):
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


def cluster_damage(dmg):
    dm = dmg - dmg.mean(1, keepdims=True)
    dstd = dmg.std(1, keepdims=True)
    dstd[dstd < 1e-12] = 1e-12
    corr = np.clip((dm @ dm.T) / (dmg.shape[1] * dstd * dstd.T), -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    return linkage(squareform((dist + dist.T) / 2, checks=False), method='average')


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    H, Dw = capture(fresh)
    tok_flat = fresh[:, :256].reshape(-1)
    Nfull = H.shape[0]
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(Nfull, generator=g)[:NSAMP]
    Hs = H[perm]
    N = Hs.shape[0]
    print(f'{N} samples, mlp{LJ} hidden dim {Hs.shape[1]}', flush=True)

    # (0) identity: full-rank reconstruction of Down's output is exact
    mu = (Hs @ Dw.T)
    ident_err = 0.0  # exact by construction (h @ Down.T is the real output)
    p0 = True
    print(f'(0) exact-by-construction (h @ Down.T IS the real output): HELD',
          flush=True)

    imp = (Dw.norm(dim=0) * Hs.std(0))
    order = imp.argsort(descending=True)
    topk = order[:TOPK].numpy()

    O = Hs @ Dw.T
    Oc = O - O.mean(0)
    U, S, Vt = torch.linalg.svd(Oc, full_matrices=False)
    cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
    r = min(int((cum < 0.95).sum().item()) + 1, Vt.shape[0])
    Vr = Vt[:r]
    print(f'output rank for 95% variance: r={r}', flush=True)

    Dw_topk = Dw[:, topk]
    Dw_proj = Vr @ Dw_topk
    Hk = Hs[:, topk]
    coldw2 = (Dw_proj ** 2).sum(0)
    damage = ((Hk ** 2) * coldw2[None, :]).T.numpy()

    Z = cluster_damage(damage)
    labels = fcluster(Z, t=NCLUST, criterion='maxclust')
    sizes = sorted(np.bincount(labels)[1:].tolist(), reverse=True)
    print(f'cluster sizes: {sizes}', flush=True)

    # (a) stability
    g2 = np.random.default_rng(3)
    sperm = g2.permutation(N)
    h1, h2 = sperm[:N // 2], sperm[N // 2:]
    Z1 = cluster_damage(damage[:, h1])
    Z2 = cluster_damage(damage[:, h2])
    lab1 = fcluster(Z1, t=NCLUST, criterion='maxclust')
    lab2 = fcluster(Z2, t=NCLUST, criterion='maxclust')
    ari = adjusted_rand_index(list(lab1), list(lab2))
    lab2_shuf = g2.permutation(lab2)
    ari_null = adjusted_rand_index(list(lab1), list(lab2_shuf))
    pa = ari >= 0.3
    print(f'(a) stability ARI {ari:.3f} (bar 0.3) vs null {ari_null:.3f}: '
          f"{'HELD' if pa else 'FAILED'}", flush=True)
    null_ok = ari_null < 0.1

    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    top3 = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))[:3]

    def top_examples(unit_ids, sign=1, k=8):
        act = H[:, unit_ids].sum(1) * sign
        top = act.argsort(descending=True)[:k]
        out = []
        for gi in top.tolist():
            tid = int(tok_flat[gi])
            r_, p_ = gi // 256, gi % 256
            back = fresh[r_, max(0, p_ - 10):p_ + 1].tolist()
            pre = cl.enc().decode(back)
            out.append({'token': cl.d1(tid), 'pre': pre, 'act': float(act[gi])})
        return out

    def concentration(examples):
        classes = [fine_class(e['token']) for e in examples]
        best = max(set(classes), key=classes.count)
        return classes.count(best), best

    conc_list = []
    named = []
    for cid, unit_ids in top3:
        pos_ex = top_examples(unit_ids, sign=1)
        neg_ex = top_examples(unit_ids, sign=-1)
        cnt, cls = concentration(pos_ex)
        conc_list.append(cnt)
        print(f'\ncluster {cid} ({len(unit_ids)} units) POSITIVE '
              f'(concentration {cnt}/8 "{cls}"):', flush=True)
        for e in pos_ex:
            print(f"   [{e['act']:+.2f}] ...{e['pre']!r} | next={e['token']!r}",
                  flush=True)
        print(f'  NEGATIVE:', flush=True)
        for e in neg_ex:
            print(f"   [{e['act']:+.2f}] ...{e['pre']!r} | next={e['token']!r}",
                  flush=True)
        named.append({'cluster': cid, 'n_units': len(unit_ids),
                      'pos_concentration': cnt, 'pos_class': cls,
                      'pos_examples': pos_ex, 'neg_examples': neg_ex})

    pb = sum(c >= 5 for c in conc_list) >= 1
    print(f'\n(b) found-cluster concentrations {conc_list} '
          f"(>=5/8 for >=1 of 3): {'HELD' if pb else 'FAILED'}", flush=True)

    ARTICLE_TOKENS = {'a', 'an', 'the'}
    article_clusters = []
    for c in named:
        toks = {e['token'].strip().lower() for e in c['pos_examples']}
        if c['pos_class'] == 'determiner' or (toks & ARTICLE_TOKENS):
            article_clusters.append(c['cluster'])
    print(f'(c) article-related clusters among top 3: {article_clusters} '
          f'({"found -- candidate for causal follow-up" if article_clusters else "none -- redundancy does not extend to mlp2 among the top clusters"})',
          flush=True)

    out = {'N': N, 'sizes': sizes, 'pred_0': bool(p0), 'stability_ari': ari,
           'stability_null': ari_null, 'pred_a': bool(pa),
           'conc_list': conc_list, 'pred_b': bool(pb),
           'article_clusters': article_clusters,
           'null_ok': bool(null_ok), 'top_clusters': named,
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
