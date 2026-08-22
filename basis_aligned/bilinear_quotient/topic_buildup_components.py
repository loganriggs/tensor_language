"""WHICH COMPONENTS BUILD THE TOPIC TRACKER — attention or MLP? §929 showed topic is assembled gradually across
the stack, with the biggest jumps early-mid. Attribute that buildup to components: for each layer, mean-ablate
that layer's ATTENTION output (replace with its global mean, removing context-varying signal) vs its MLP output,
re-run, and measure the DROP in the final (L15) topic-decodability (topics + probe fixed from the clean run).
The component whose ablation drops topic-decode most is the one carrying topic into the final representation.
Tests the account that ATTENTION is the context aggregator/courier for content (§862/§871) while MLPs read/emit.

REGISTERED PREDICTIONS:
  (0) SANITY: clean L15 topic-decode reproduces §929 (~0.85, base ~0.14); ablating a component in the readout
      layers L16-17 barely changes L15 topic-decode (measured upstream of them -> ~0 drop, a placement control).
  (a) ATTENTION BUILDS TOPIC: summed over layers, ATTENTION-ablation drops L15 topic-decode MORE than
      MLP-ablation, and the largest attention drops fall in the early-mid layers (L1-L9) where §929 showed the
      biggest buildup -> attention is the topic aggregator; MLP contributes less to assembling topic;
  (b) report per-(layer,component) topic-decode drop; note if any single MLP layer is a large contributor."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'topic_buildup_components_results.json'
CONTENT_L = 15; NLAYER = 18; NEVAL = 160; SEQ = 256; RTOK = 64; RPOS = 32; K = 32; RIDGE = 1e2
ABL = {'L': -1, 'comp': None, 'mean': None}  # active ablation


def dec():
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); return lambda i: enc.decode([int(i)])


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def mk_hook(L, comp):
    def h(mo, i_, o_):
        if ABL['L'] != L or ABL['comp'] != comp: return o_
        y = o_[0] if isinstance(o_, tuple) else o_
        mv = ABL['mean'][(L, comp)].to(y.dtype)
        ny = mv.view(1, 1, D).expand_as(y).clone()
        return (ny,) + tuple(o_[1:]) if isinstance(o_, tuple) else ny
    return h


def forward_capL(idx, capL=CONTENT_L):
    cap = {}
    def ch(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[capL].register_forward_hook(ch)
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    hh.remove(); return cap['r']


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
    return a


def content_of(R, toks, pos, Ucp=None, g=None):
    if Ucp is None:
        Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
        Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    return (R-g) - ((R-g)@Ucp)@Ucp.T, Ucp, g


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    # --- pass 1: capture each submodule's global-mean output + clean L15 residual ---
    sums = {(L, c): torch.zeros(D, device=DEV) for L in range(NLAYER) for c in ('attn', 'mlp')}
    cnt = 0; caps = {}
    def cap_hook(L, comp):
        def h(mo, i_, o_):
            y = o_[0] if isinstance(o_, tuple) else o_
            sums[(L, comp)] += y.reshape(-1, D).sum(0)
        return h
    hs = []
    for L in range(NLAYER):
        hs.append(m.transformer.h[L].attn.register_forward_hook(cap_hook(L, 'attn')))
        hs.append(m.transformer.h[L].mlp.register_forward_hook(cap_hook(L, 'mlp')))
    R15c = []
    for i in range(0, nb, 4):
        idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous()
        R15c.append(forward_capL(idx).reshape(-1, D)); cnt += idx.shape[0]*idx.shape[1]
    for h in hs: h.remove()
    ABL['mean'] = {k: (v/cnt) for k, v in sums.items()}
    Rc = torch.cat(R15c, 0)
    contentc, Ucp, g = content_of(Rc, toks, pos)
    cn = contentc/(contentc.norm(dim=1, keepdim=True)+1e-9); topic = kmeans(cn, K).cpu().numpy()
    # fixed probe on clean L15 content (train/test)
    n = contentc.shape[0]; rng = np.random.RandomState(0); perm = rng.permutation(n); ntr = int(0.7*n)
    a_i, b_i = perm[:ntr], perm[ntr:]
    Y = torch.zeros(len(a_i), K, device=DEV); Y[torch.arange(len(a_i)), torch.tensor(topic[a_i], device=DEV)] = 1.0
    A = contentc[a_i].T @ contentc[a_i] + RIDGE*torch.eye(D, device=DEV); W = torch.linalg.solve(A, contentc[a_i].T @ Y)
    def score(content):
        return float((content[b_i] @ W).argmax(1).cpu().numpy().__eq__(topic[b_i]).mean())
    clean_acc = score(contentc); base = float(np.bincount(topic, minlength=K).max()/len(topic))
    # --- pass 2: per-(layer,comp) mean-ablation, measure L15 topic-decode drop ---
    ah = []
    for L in range(NLAYER):
        ah.append(m.transformer.h[L].attn.register_forward_hook(mk_hook(L, 'attn')))
        ah.append(m.transformer.h[L].mlp.register_forward_hook(mk_hook(L, 'mlp')))
    def acc_with(L, comp):
        ABL['L'] = L; ABL['comp'] = comp; RR = []
        for i in range(0, nb, 4):
            idx = blocks[i:i+4].to(DEV)[:, :-1].contiguous(); RR.append(forward_capL(idx).reshape(-1, D))
        ABL['L'] = -1; ABL['comp'] = None
        content, _, _ = content_of(torch.cat(RR, 0), toks, pos, Ucp, g); return score(content)
    drops = {'attn': {}, 'mlp': {}}
    for L in range(NLAYER):
        for comp in ('attn', 'mlp'):
            a = acc_with(L, comp); drops[comp][str(L)] = round(clean_acc - a, 4)
        print(f"L{L:>2}: attn-drop {drops['attn'][str(L)]:+.4f}  mlp-drop {drops['mlp'][str(L)]:+.4f}", flush=True)
    for h in ah: h.remove()
    sum_attn = round(sum(drops['attn'].values()), 4); sum_mlp = round(sum(drops['mlp'].values()), 4)
    attn_early = round(sum(drops['attn'][str(L)] for L in range(1, 10)), 4)
    out = {'clean_L15_topic_acc': round(clean_acc, 4), 'base_rate': round(base, 4), 'K': K,
           'attn_drops': drops['attn'], 'mlp_drops': drops['mlp'],
           'sum_attn_drop': sum_attn, 'sum_mlp_drop': sum_mlp, 'attn_early_L1_9_drop': attn_early,
           'readout_L16_17_attn_drop': round(drops['attn']['16']+drops['attn']['17'], 4)}
    out['pred_a_attention_builds_topic'] = bool(sum_attn > sum_mlp and attn_early > 0)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"clean {clean_acc:.3f} base {base:.3f} | SUM attn-drop {sum_attn} vs mlp-drop {sum_mlp} | attn early(L1-9) {attn_early}", flush=True)
    print(f"(a) attention builds topic (sum attn-drop > mlp, early-concentrated): {out['pred_a_attention_builds_topic']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
