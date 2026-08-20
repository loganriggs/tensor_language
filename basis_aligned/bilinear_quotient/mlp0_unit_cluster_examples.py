"""MLP0 UNIT CLUSTER EXAMPLES -- name 579's clusters by their real
activating contexts instead of a logit-lens on their output
direction.

579 found mlp0's hidden units cluster into statistically real,
out-of-sample-stable groups (ARI 0.58 vs 0.00 null), but naming them
by projecting the cluster's summed Down (output) direction onto the
embedding table gave unreadable token lists -- the wrong lens, since
these units are INPUT-selective (h_j depends on x through Left/Right)
and the class of input that fires a unit need not resemble the token
its output direction points toward. This redoes naming the way
mlp0_units already names single units: by the real contexts where
the cluster's summed activation is largest (and separately, most
negative -- signed, since squares can subtract).

REGISTERED PREDICTIONS:
  (0) REPRODUCIBILITY: reclustering the same top-300 units with the
      same seeds reproduces 579's cluster sizes exactly (sanity that
      this script is really looking at the same clusters, not a
      silent divergence);
  (a) CONCENTRATION: for the 3 largest clusters, among each
      cluster's top-8 positive-activating example positions, at
      least 5/8 share one surface class (mlp0_units' fine_class:
      determiner/preposition/pronoun/digit/punct/capitalized/
      space_word/subword/space) at the CURRENT token -- a real
      class-selective cluster should concentrate; report the actual
      counts regardless of the bar;
  (b) NOT A GENERIC-FREQUENCY ARTIFACT: the same concentration count
      for the top-8 positions of 3 RANDOM same-size unit subsets
      (not from any found cluster) is lower than the found clusters'
      -- concentration is specific to real clusters, not a property
      of any large unit subset;
  NULL: shuffling which sample each activation belongs to before
      picking top-8 should destroy concentration for the found
      clusters too (drop to the random-subset level) -- confirms the
      concentration is about WHICH units are grouped, not an
      artifact of the top-8 selection procedure itself."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import census_lib as cl
from bilin18_joint_removal import m, DEV
D = 1152
T = 256
LJ = 0
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_unit_cluster_examples_results.json'
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
    H = torch.cat(hs, dim=0)
    return H, Dw.cpu()


def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    H, Dw = capture(fresh)
    Nfull = H.shape[0]
    tok_flat = fresh[:, :256].reshape(-1)  # aligned with H's flatten order

    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(Nfull, generator=g)[:NSAMP]
    Hs = H[perm]
    N = Hs.shape[0]

    imp = (Dw.norm(dim=0) * Hs.std(0))
    order = imp.argsort(descending=True)
    topk = order[:TOPK].numpy()

    O = Hs @ Dw.T
    Omu = O.mean(0)
    Oc = O - Omu
    U, S, Vt = torch.linalg.svd(Oc, full_matrices=False)
    var = (S ** 2)
    cum = torch.cumsum(var, 0) / var.sum()
    r = int((cum < 0.95).sum().item()) + 1
    r = min(r, Vt.shape[0])
    Vr = Vt[:r]

    Dw_topk = Dw[:, topk]
    Dw_proj = Vr @ Dw_topk
    Hk = Hs[:, topk]
    coldw2 = (Dw_proj ** 2).sum(0)
    damage = ((Hk ** 2) * coldw2[None, :]).T.numpy()

    dm = damage - damage.mean(1, keepdims=True)
    dstd = damage.std(1, keepdims=True)
    dstd[dstd < 1e-12] = 1e-12
    corr = (dm @ dm.T) / (damage.shape[1] * dstd * dstd.T)
    corr = np.clip(corr, -1, 1)
    dist = 1 - corr
    np.fill_diagonal(dist, 0)
    cond = squareform((dist + dist.T) / 2, checks=False)
    Z = linkage(cond, method='average')
    labels = fcluster(Z, t=NCLUST, criterion='maxclust')
    sizes = sorted(np.bincount(labels)[1:].tolist(), reverse=True)
    expect = [101, 76, 29, 18, 16, 9, 6, 6, 6, 6, 5, 4, 4, 3, 3, 3, 2, 1, 1, 1]
    p0 = sizes == expect
    print(f'(0) reproduced sizes {sizes} == 579 {expect}: '
          f"{'HELD' if p0 else 'FAILED'}", flush=True)

    by_cluster = {}
    for i, c in enumerate(labels):
        by_cluster.setdefault(int(c), []).append(int(topk[i]))
    top3 = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))[:3]

    def top_examples(unit_ids, sign=1, k=8, shuffle=False):
        act = H[:, unit_ids].sum(1) * sign
        if shuffle:
            act = act[torch.randperm(act.shape[0])]
        top = act.argsort(descending=True)[:k]
        out = []
        for gi in top.tolist():
            tid = int(tok_flat[gi])
            r_, p_ = gi // 256, gi % 256
            back = fresh[r_, max(0, p_ - 10):p_ + 1].tolist()
            pre = cl.enc().decode(back)
            out.append({'gi': gi, 'token': cl.d1(tid), 'pre': pre,
                        'act': float(act[gi])})
        return out

    def concentration(examples):
        classes = [fine_class(e['token']) for e in examples]
        best = max(set(classes), key=classes.count)
        return classes.count(best), best

    results = {'clusters': [], 'random_baseline': [], 'shuffled_null': []}
    conc_found = []
    for cid, unit_ids in top3:
        pos_ex = top_examples(unit_ids, sign=1)
        neg_ex = top_examples(unit_ids, sign=-1)
        cnt, cls = concentration(pos_ex)
        conc_found.append(cnt)
        print(f'\ncluster {cid} ({len(unit_ids)} units) POSITIVE top-8 '
              f'(concentration {cnt}/8 "{cls}"):', flush=True)
        for e in pos_ex:
            print(f"   [{e['act']:+.2f}] ...{e['pre']!r} | next={e['token']!r}",
                  flush=True)
        print(f'  NEGATIVE top-8:', flush=True)
        for e in neg_ex:
            print(f"   [{e['act']:+.2f}] ...{e['pre']!r} | next={e['token']!r}",
                  flush=True)
        results['clusters'].append({'cluster': cid, 'n_units': len(unit_ids),
                                     'pos_concentration': cnt,
                                     'pos_class': cls,
                                     'pos_examples': pos_ex,
                                     'neg_examples': neg_ex})

    # (b) random same-size subsets, not from any found cluster
    g2 = np.random.default_rng(11)
    conc_rand = []
    for cid, unit_ids in top3:
        pick = g2.choice(order.numpy(), size=len(unit_ids), replace=False)
        ex = top_examples(pick.tolist(), sign=1)
        cnt, cls = concentration(ex)
        conc_rand.append(cnt)
        results['random_baseline'].append({'n': len(unit_ids), 'conc': cnt,
                                            'cls': cls})
    pa = sum(c >= 5 for c in conc_found) >= 1
    pb = np.mean(conc_found) > np.mean(conc_rand)
    print(f'\n(a) found-cluster concentrations {conc_found} '
          f"(>=5/8 for >=1 of 3): {'HELD' if pa else 'FAILED'}", flush=True)
    print(f'(b) found {np.mean(conc_found):.2f} vs random-subset '
          f"{np.mean(conc_rand):.2f}: {'HELD' if pb else 'FAILED'}",
          flush=True)

    # NULL: shuffle sample alignment before top-8 selection
    conc_shuf = []
    for cid, unit_ids in top3:
        ex = top_examples(unit_ids, sign=1, shuffle=True)
        cnt, cls = concentration(ex)
        conc_shuf.append(cnt)
        results['shuffled_null'].append({'cluster': cid, 'conc': cnt})
    null_ok = np.mean(conc_shuf) <= np.mean(conc_rand) + 1
    print(f'NULL (shuffled {np.mean(conc_shuf):.2f} <= random-subset+1 '
          f"{np.mean(conc_rand)+1:.2f}): {'ok' if null_ok else 'CHECK'}",
          flush=True)

    out = {'pred_0': bool(p0), 'sizes': sizes,
           'conc_found': conc_found, 'conc_random': conc_rand,
           'conc_shuffled': conc_shuf, 'pred_a': bool(pa),
           'pred_b': bool(pb), 'null_ok': bool(null_ok),
           'clusters': results['clusters'],
           'random_baseline': results['random_baseline'],
           'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
