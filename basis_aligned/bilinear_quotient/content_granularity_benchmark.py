"""DOES NAMING CONTENT AS A FINE-GRAINED TOPIC HIERARCHY (§927) IMPROVE THE UNDERSTANDING BENCHMARK? Causal,
held-out test tied to the benchmark's CONTENT term. At the content layer (L15), decompose the residual into a
STRUCTURED part (token+position+next-class subspaces = grammar, kept exactly) and a CONTENT part (the
remainder). Replace the CONTENT part with its nearest-of-K topic-centroid reconstruction, re-inject, and run the
rest of the model. Recovery = (loss_ablate_content - loss_K) / (loss_ablate_content - loss_full), where
loss_ablate_content = replace content by its single global mean (K=1). Sweep K to see how much of the content
machine a topic-centroid stand-in captures as topics get finer. Centroids FIT ON TRAIN rows, evaluated on
HELD-OUT rows (assign test points to nearest train centroid) so recovery is not same-data overfit.

Ties §927 (content = coarse->fine topic hierarchy) to the tracked understanding benchmark: if recovery keeps
rising with K, the content is genuinely fine-grained subject matter and a topic-centroid stand-in is a real
(partial) account of it; the value at large K is the content term's ceiling for this stand-in family.

REGISTERED PREDICTIONS:
  (0) SANITY: full model loss < content-ablated loss (content matters); K=1 recovery == 0 by construction;
      a SHUFFLED-assignment null (assign test points to a random centroid) recovers ~0 (<0.03).
  (a) GRANULARITY HELPS: held-out content recovery INCREASES monotonically with K (8 < 32 < 128) and does not
      plateau early -> content is fine-grained subject matter (consistent with the §927 topic->subtopic
      hierarchy), and a topic-centroid stand-in recovers a growing fraction of the content machine;
  (b) report the recovery(K) curve + the shuffled null; the large-K value is the topic-centroid content ceiling.
"""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_granularity_benchmark_results.json'
CONTENT_L = 15; NEVAL = 300; SEQ = 256; RTOK = 64; RPOS = 32; RCLASS = 8
KS = [1, 8, 32, 128]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
SUB = {'on': False, 'newout': None}  # substitution state for block-15 output


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def classify(s):
    t = s.strip()
    if t == '' or not t[0].isalnum(): return 'punct'
    if t[0].isdigit(): return 'number'
    low = t.lower()
    if low in DET: return 'det'
    if low in PREP: return 'prep'
    if low in CONJ: return 'conj'
    if low in PRON: return 'pron'
    if t[0].isupper(): return 'cap'
    return 'word'


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def sub_hook(mo, i_, o_):
    """Replace block-CONTENT_L output with SUB['newout'] (shape matches) when on."""
    if not SUB['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_
    ny = SUB['newout'].to(y.dtype)
    return (ny,) + tuple(o_[1:]) if isinstance(o_, tuple) else ny


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capL(idx):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[CONTENT_L].register_forward_hook(h); forward_logits(idx); hh.remove(); return cap['r']


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def kmeans_fit(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return c


@torch.no_grad()
def loss_on(test_blocks, recon_content=None, ustruct=None, g=None, cn_norm=None, centroids=None,
            content_centroid_raw=None, shuffle=False, seed=0):
    """Run the model on test_blocks. If centroids given, substitute the CONTENT part of block-15 output with
    the nearest (train) centroid's raw content vector; else run clean (baseline)."""
    tot_lp = []; ntok = 0; rng = np.random.RandomState(seed)
    for i in range(0, test_blocks.shape[0], 4):
        bb = test_blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        R = capL(idx)  # (b, SEQ-1, D)  clean block-15 output
        if centroids is not None:
            b, T, _ = R.shape; Rf = R.reshape(-1, D)
            struct = g + ((Rf - g) @ ustruct) @ ustruct.T
            content = Rf - struct
            cnf = content / (content.norm(dim=1, keepdim=True) + 1e-9)
            assign = torch.cdist(cnf, cn_norm).argmin(1)
            if shuffle:
                assign = torch.tensor(rng.randint(0, centroids.shape[0], size=assign.shape[0]), device=DEV)
            new_content = content_centroid_raw[assign]
            newRf = struct + new_content
            SUB['newout'] = newRf.reshape(b, T, D); SUB['on'] = True
        else:
            SUB['on'] = False
        lg = forward_logits(idx).float(); SUB['on'] = False
        lp = F.log_softmax(lg, -1); tf = tgt.reshape(-1)
        lpf = lp.reshape(-1, lp.shape[-1])
        tot_lp.append((-lpf[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
    w = np.concatenate(tot_lp); return float(w.mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    ntr = int(0.6 * nb); tr = blocks[:ntr]; te = blocks[ntr:]
    Str = tr.cpu().numpy()
    # --- fit structured subspace + content centroids on TRAIN ---
    Rs = []
    for i in range(0, ntr, 4): Rs.append(capL(tr[i:i+4].to(DEV)[:, :-1].contiguous()).reshape(-1, D))
    Rtr = torch.cat(Rs, 0)
    toks = Str[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (ntr, SEQ-1)).reshape(-1)
    nxtc = np.full_like(Str[:, :-1], -1); nxtc[:, :-1] = Str[:, 1:-1]
    nxtcls = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxtc.reshape(-1)])
    Utok, g = mean_subspace(Rtr, toks, RTOK); Upos, _ = mean_subspace(Rtr, pos.astype(np.int64), RPOS)
    Uclass, _ = mean_subspace(Rtr, nxtcls, RCLASS)
    Ustruct = torch.linalg.svd(torch.cat([Utok, Upos, Uclass], 1), full_matrices=False)[0][:, :RTOK+RPOS+RCLASS].contiguous()
    content_tr = (Rtr - g) - ((Rtr - g) @ Ustruct) @ Ustruct.T
    cntr = content_tr / (content_tr.norm(dim=1, keepdim=True) + 1e-9)
    hh = m.transformer.h[CONTENT_L].register_forward_hook(sub_hook)
    loss_full = loss_on(te)
    out = {'loss_full': round(loss_full, 4), 'ntrain_rows': ntr, 'neval_rows': nb - ntr, 'recovery': {}, 'loss_K': {}}
    loss_ablate = None
    for K in KS:
        cen = kmeans_fit(cntr, K, seed=0) if K > 1 else cntr.mean(0, keepdim=True)/(cntr.mean(0, keepdim=True).norm()+1e-9)
        # raw content vector per centroid = mean of raw content over train pts assigned to it
        assign_tr = torch.cdist(cntr, cen).argmin(1)
        raw = torch.stack([content_tr[assign_tr == j].mean(0) if (assign_tr == j).any() else torch.zeros(D, device=DEV) for j in range(cen.shape[0])], 0)
        lK = loss_on(te, ustruct=Ustruct, g=g, cn_norm=cen, centroids=cen, content_centroid_raw=raw)
        out['loss_K'][str(K)] = round(lK, 4)
        if K == 1: loss_ablate = lK
        rec = (loss_ablate - lK) / (loss_ablate - loss_full + 1e-9)
        out['recovery'][str(K)] = round(float(rec), 4)
        print(f"K={K:>4}: loss {lK:.4f}  recovery {rec:+.4f}", flush=True)
    # shuffled-assignment null at largest K
    K = KS[-1]; cen = kmeans_fit(cntr, K, seed=0); assign_tr = torch.cdist(cntr, cen).argmin(1)
    raw = torch.stack([content_tr[assign_tr == j].mean(0) if (assign_tr == j).any() else torch.zeros(D, device=DEV) for j in range(K)], 0)
    lsh = loss_on(te, ustruct=Ustruct, g=g, cn_norm=cen, centroids=cen, content_centroid_raw=raw, shuffle=True, seed=1)
    out['loss_shuffled_null'] = round(lsh, 4)
    out['recovery_shuffled_null'] = round(float((loss_ablate - lsh) / (loss_ablate - loss_full + 1e-9)), 4)
    hh.remove()
    recs = [out['recovery'][str(k)] for k in KS]
    out['pred_a_granularity_helps'] = bool(recs[1] < recs[2] < recs[3] and out['recovery_shuffled_null'] < 0.03)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"shuffled null recovery {out['recovery_shuffled_null']:+.4f}", flush=True)
    print(f"(a) granularity helps (recovery rises 8<32<128, null~0): {out['pred_a_granularity_helps']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
