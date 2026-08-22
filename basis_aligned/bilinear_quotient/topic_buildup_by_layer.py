"""WHERE ACROSS THE 18 LAYERS IS THE TOPIC TRACKER BUILT? (bottom-up, no-skipping). The content machine is a
topic tracker (§866) living in the high-rank content residual; §912 traced the GRAMMAR (class) carrier layer by
layer. Do the same for TOPIC: at every layer's output, strip token+position, cluster the leftover content into
K topics on TRAIN rows, and measure how well those topic labels are DECODABLE (held-out) from that layer's
content residual. This locates the depth at which topic information rises — the construction curve of the
content machine, to sit alongside the grammar-carrier curve.

Also report NEWLY-WRITTEN topic each layer: the increase in topic-decodability from layer L-1 to L (how much
each block adds), to see whether topic is built in the front (like grammar) or accumulated by mid/late
attention aggregation (§862/§871 predict long-range mid/late aggregation).

REGISTERED PREDICTIONS:
  (0) SANITY: topic decodability at the late content layer (L15) is well above a shuffled-label null;
  (a) LATE/GRADUAL BUILD: topic decodability RISES through the middle-to-late layers and is LOW in the front
      (L0-2), mirroring long-range aggregation (§871) and contrasting with grammar, which is written in the
      front (mlp0) -> topic is built later and more gradually than grammar;
  (b) report per-layer topic decodability + per-layer increment + the front-vs-late contrast with grammar."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_buildup_by_layer_results.json'
NLAYER = 18; NEVAL = 240; SEQ = 256; RTOK = 64; RPOS = 32; K = 32; RIDGE = 1e2
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


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


def forward_capture_all(idx):
    """Return list of per-layer outputs (each (b,T,D)) on CPU-float."""
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None; outs = []
    for blk in m.transformer.h:
        x, v1 = blk(x, v1, x0); outs.append(x.detach().float())
    return outs


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        if t < 0: continue
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a, c


def content_of(R, toks, pos):
    Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R-g) - ((R-g)@Ucp)@Ucp.T
    return content


def acc(F_, y, ncls, seed=0):
    n = F_.shape[0]; rng = np.random.RandomState(seed); idx = rng.permutation(n); ntr = int(0.7*n); a, b = idx[:ntr], idx[ntr:]
    Y = torch.zeros(len(a), ncls, device=DEV); Y[torch.arange(len(a)), torch.tensor(y[a], device=DEV)] = 1.0
    A = F_[a].T @ F_[a] + RIDGE*torch.eye(F_.shape[1], device=DEV); W = torch.linalg.solve(A, F_[a].T @ Y)
    return float(((F_[b] @ W).argmax(1).cpu().numpy() == y[b]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    # capture all layers (CPU to fit memory), then process per layer
    perlayer = [[] for _ in range(NLAYER)]
    for i in range(0, nb, 4):
        outs = forward_capture_all(blocks[i:i+4].to(DEV)[:, :-1].contiguous())
        for L in range(NLAYER): perlayer[L].append(outs[L].reshape(-1, D).cpu())
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    # define topic labels from the LATE content layer (L15) so all layers are scored against the same targets
    R15 = torch.cat(perlayer[15], 0).to(DEV)
    c15 = content_of(R15, toks, pos); cn = c15/(c15.norm(dim=1, keepdim=True)+1e-9)
    topic, _ = kmeans(cn, K); topic = topic.cpu().numpy()
    rng = np.random.RandomState(0); tsh = topic.copy(); rng.shuffle(tsh)
    base = float(np.bincount(topic, minlength=K).max()/len(topic))
    out = {'K': K, 'base_rate': round(base, 4), 'topic_target_layer': 15, 'per_layer': {}, 'increment': {}}
    prev = None
    for L in range(NLAYER):
        RL = torch.cat(perlayer[L], 0).to(DEV)
        cL = content_of(RL, toks, pos)
        a = acc(cL, topic, K)
        out['per_layer'][str(L)] = round(a, 4)
        if prev is not None: out['increment'][str(L)] = round(a - prev, 4)
        prev = a
        del RL, cL; torch.cuda.empty_cache()
        print(f"L{L:>2}: topic-decode {a:.4f}", flush=True)
    a_sh = acc(content_of(R15, toks, pos), tsh, K)
    out['shuffled_null_L15'] = round(a_sh, 4)
    per = [out['per_layer'][str(L)] for L in range(NLAYER)]
    front = float(np.mean(per[0:3])); late = float(np.mean(per[13:16]))
    out['front_L0_2_mean'] = round(front, 4); out['late_L13_15_mean'] = round(late, 4)
    out['pred_a_late_gradual'] = bool(late > front + 0.1 and per[15] > a_sh + 0.1)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"front(L0-2) {front:.3f} vs late(L13-15) {late:.3f} | shuffled null {a_sh:.3f}", flush=True)
    print(f"(a) topic built late/gradual (late>>front, above null): {out['pred_a_late_gradual']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
