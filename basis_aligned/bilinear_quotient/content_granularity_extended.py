"""IS THE CONTENT MACHINE A FINITE TOPIC SET OR A CONTINUUM? §928 showed a topic-centroid stand-in for the L15
content part recovers only 0.8/2.4/6.3% of the content machine's loss at K=8/32/128 (monotonic, no plateau).
Extend the K sweep to 512 and 1024 (with more training rows so large-K centroids stay well-populated) to
distinguish two hypotheses: (i) FINITE topic set -> recovery PLATEAUS at some K (a finite number of topics
captures content); (ii) CONTINUUM (§873/§874) -> recovery keeps RISING with no plateau, because content is a
smooth high-rank space where you would need ~a centroid per context. Same causal, held-out protocol as §928
(keep token+position+next-class exact, replace the content part with nearest-of-K train centroid, run 16-17).

REGISTERED PREDICTIONS:
  (0) SANITY: reproduces §928 at K=128 (recovery ~0.06); shuffled-assignment null recovers <=0 (worse than mean).
  (a) CONTINUUM: recovery keeps RISING from 128 -> 512 -> 1024 with NO plateau (each step still adds recovery),
      and remains modest (<0.20 even at 1024) -> content is a high-rank continuum, not a finite topic set;
      to recover a large fraction you would need ~a centroid per context.
  (b) report the extended recovery(K) curve + per-step increments; if recovery flattens (increment -> ~0), that
      would instead support a finite effective topic count (report the plateau K)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_granularity_extended_results.json'
CONTENT_L = 15; NEVAL = 500; SEQ = 256; RTOK = 64; RPOS = 32; RCLASS = 8
KS = [1, 128, 512, 1024]
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
SUB = {'on': False, 'newout': None}


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
    if not SUB['on']: return o_
    y = o_[0] if isinstance(o_, tuple) else o_; ny = SUB['newout'].to(y.dtype)
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


def kmeans_fit(X, k, iters=20, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed)
    c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return c


@torch.no_grad()
def loss_on(test_blocks, ustruct=None, g=None, cn_norm=None, content_centroid_raw=None, shuffle=False, seed=0):
    tot = []; rng = np.random.RandomState(seed)
    for i in range(0, test_blocks.shape[0], 4):
        bb = test_blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        R = capL(idx)
        if cn_norm is not None:
            b, T, _ = R.shape; Rf = R.reshape(-1, D)
            struct = g + ((Rf - g) @ ustruct) @ ustruct.T; content = Rf - struct
            cnf = content / (content.norm(dim=1, keepdim=True) + 1e-9)
            assign = torch.cdist(cnf, cn_norm).argmin(1)
            if shuffle: assign = torch.tensor(rng.randint(0, cn_norm.shape[0], size=assign.shape[0]), device=DEV)
            SUB['newout'] = (struct + content_centroid_raw[assign]).reshape(b, T, D); SUB['on'] = True
        else:
            SUB['on'] = False
        lg = forward_logits(idx).float(); SUB['on'] = False
        lp = F.log_softmax(lg, -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        tot.append((-lpf[torch.arange(tf.shape[0], device=DEV), tf]).cpu().numpy())
    return float(np.concatenate(tot).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    ntr = int(0.6 * nb); tr = blocks[:ntr]; te = blocks[ntr:]; Str = tr.cpu().numpy()
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
    out = {'loss_full': round(loss_full, 4), 'ntrain_rows': ntr, 'neval_rows': nb - ntr, 'npts_train': int(cntr.shape[0]),
           'recovery': {}, 'loss_K': {}, 'increment': {}}
    loss_ablate = None; prev = None
    for K in KS:
        if K == 1:
            cen = cntr.mean(0, keepdim=True); cen = cen/(cen.norm()+1e-9)
        else:
            cen = kmeans_fit(cntr, K, seed=0)
        assign_tr = torch.cdist(cntr, cen).argmin(1)
        raw = torch.stack([content_tr[assign_tr == j].mean(0) if (assign_tr == j).any() else torch.zeros(D, device=DEV) for j in range(cen.shape[0])], 0)
        lK = loss_on(te, ustruct=Ustruct, g=g, cn_norm=cen, content_centroid_raw=raw)
        out['loss_K'][str(K)] = round(lK, 4)
        if K == 1: loss_ablate = lK
        rec = float((loss_ablate - lK) / (loss_ablate - loss_full + 1e-9)); out['recovery'][str(K)] = round(rec, 4)
        if prev is not None: out['increment'][str(K)] = round(rec - prev, 4)
        prev = rec
        print(f"K={K:>4}: loss {lK:.4f}  recovery {rec:+.4f}", flush=True)
    # shuffled null at largest K
    K = KS[-1]; cen = kmeans_fit(cntr, K, seed=0); assign_tr = torch.cdist(cntr, cen).argmin(1)
    raw = torch.stack([content_tr[assign_tr == j].mean(0) if (assign_tr == j).any() else torch.zeros(D, device=DEV) for j in range(K)], 0)
    lsh = loss_on(te, ustruct=Ustruct, g=g, cn_norm=cen, content_centroid_raw=raw, shuffle=True, seed=1)
    out['recovery_shuffled_null'] = round(float((loss_ablate - lsh) / (loss_ablate - loss_full + 1e-9)), 4)
    hh.remove()
    incs = [out['increment'][str(k)] for k in KS[1:]]
    out['pred_a_continuum'] = bool(all(i > 0.003 for i in incs) and out['recovery'][str(KS[-1])] < 0.20 and out['recovery_shuffled_null'] <= 0.0)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"increments {out['increment']} | shuffled null {out['recovery_shuffled_null']:+.4f}", flush=True)
    print(f"(a) continuum (recovery keeps rising, no plateau, still modest): {out['pred_a_continuum']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
