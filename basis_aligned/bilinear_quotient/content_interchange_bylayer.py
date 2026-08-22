"""Is the CONTENT variable causally addressable WHERE it is computed (the MIDDLE, §940), not just at L15? §894/§959
did content interchange at L15 (readout-adjacent). Verify content causality at MIDDLE layers by patching the
content/topic subspace base<-source at L8, L11, L15 and measuring the class-controlled topic shift (gain on
source-topic MINUS base-topic distinctive tokens, §959's clean metric). This extends causal abstraction into the
middle, no-skipping.

REGISTERED PREDICTIONS:
  (0) SANITY: random-subspace patch gives topic_net ~0 at every layer.
  (a) CONTENT CAUSAL IN THE MIDDLE: content-subspace interchange gives positive class-controlled topic_net (>>
      random) at L8 and L11 (where content is computed, §940), not only at L15 -> the content variable is causally
      addressable throughout the middle; report whether the effect grows/stable across L8->L11->L15;
  (b) report content-patch vs random-patch topic_net per layer."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from collections import Counter

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_interchange_bylayer_results.json'
NEVAL = 200; SEQ = 256; RTOK = 64; RPOS = 32; K = 12; RCONTENT = 24; QP = 200
LAYERS = [8, 11, 15]
PATCH = {'on': False, 'vec': None, 'U': None, 'L': -1}


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def patch_hook_factory(L):
    def h(mo, i_, o_):
        if not PATCH['on'] or PATCH['L'] != L: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; y = y.clone(); U = PATCH['U']
        b = y[:, QP, :]; y[:, QP, :] = b - (b @ U) @ U.T + PATCH['vec']
        return (y,) + tuple(o_[1:]) if isinstance(o_, tuple) else y
    return h


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return readout(x)


@torch.no_grad()
def capL(idx, L):
    cap = {}
    def h(mo, i_, o_): cap['r'] = (o_[0] if isinstance(o_, tuple) else o_).detach().float()
    hh = m.transformer.h[L].register_forward_hook(h); forward_logits(idx); hh.remove(); return cap['r']


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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    toks = S[:, :-1].reshape(-1); pos = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    tgt = S[:, 1:].reshape(-1); base_ct = Counter(tgt[tgt >= 0]); Nn = int((tgt >= 0).sum())
    gd = torch.Generator(device=DEV).manual_seed(0); Urand = torch.linalg.qr(torch.randn(D, RCONTENT, generator=gd, device=DEV))[0]
    # per-layer content subspace + topic labels + per-seq QP residual
    layer_data = {}
    hooks_all = {L: m.transformer.h[L].register_forward_hook(patch_hook_factory(L)) for L in LAYERS}
    for L in LAYERS:
        Rs = []
        for i in range(0, nb, 4): Rs.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), L).reshape(-1, D))
        R = torch.cat(Rs, 0)
        Utok, g = mean_subspace(R, toks, RTOK); Upos, _ = mean_subspace(R, pos.astype(np.int64), RPOS)
        Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
        content = (R-g) - ((R-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
        tlab = kmeans(cn, K).cpu().numpy(); Utopic, _ = mean_subspace(content, tlab, RCONTENT)
        tl = tlab.reshape(nb, SEQ-1)
        distinct = {}
        for j in range(K):
            mk = tl.reshape(-1) == j; nc = Counter()
            for t in np.unique(tgt[mk]):
                if t < 0: continue
                c = int((tgt[mk] == t).sum())
                if c < 4: continue
                nc[t] = (c/max(mk.sum(),1))/((base_ct.get(t,0)+1)/Nn)
            distinct[j] = set(int(t) for t,_ in nc.most_common(30))
        qv = []  # capture per-seq QP residual in chunks (avoid full-batch lm_head OOM)
        for i in range(0, nb, 4): qv.append(capL(blocks[i:i+4].to(DEV)[:, :-1].contiguous(), L)[:, QP, :])
        layer_data[L] = {'Utopic': Utopic, 'tl': tl, 'distinct': distinct, 'qvec': torch.cat(qv, 0)}
    pairs = [(i, (i+nb//2) % nb) for i in range(nb)]
    def dset(distinct, topic):
        s = distinct.get(topic, set()); return torch.tensor(sorted(s), device=DEV) if s else None
    out = {'QP': QP, 'by_layer': {}}
    for L in LAYERS:
        ld = layer_data[L]; agg = {'content': 0.0, 'random': 0.0, 'n': 0}
        for (bi, si) in pairs:
            st = int(ld['tl'][si, QP]); bt = int(ld['tl'][bi, QP])
            if st == bt: continue
            Ds = dset(ld['distinct'], st); Db = dset(ld['distinct'], bt)
            if Ds is None or Db is None: continue
            idx = blocks[bi:bi+1, :SEQ].to(DEV)[:, :-1].contiguous(); src = ld['qvec'][si]
            PATCH['on'] = False; lp0 = F.log_softmax(forward_logits(idx).float()[0, QP], -1)
            bs = float(lp0[Ds].mean()); bb = float(lp0[Db].mean())
            for name, U in [('content', ld['Utopic']), ('random', Urand)]:
                PATCH['U'] = U; PATCH['vec'] = (src @ U) @ U.T; PATCH['L'] = L; PATCH['on'] = True
                lp = F.log_softmax(forward_logits(idx).float()[0, QP], -1); PATCH['on'] = False
                agg[name] += (float(lp[Ds].mean())-bs) - (float(lp[Db].mean())-bb)
            agg['n'] += 1
        out['by_layer'][str(L)] = {'content_topic_net': round(agg['content']/max(agg['n'],1), 4),
                                   'random_topic_net': round(agg['random']/max(agg['n'],1), 4), 'n_pairs': agg['n']}
        print(f"L{L}: content topic_net {out['by_layer'][str(L)]['content_topic_net']:+.4f} | random {out['by_layer'][str(L)]['random_topic_net']:+.4f}", flush=True)
    for h in hooks_all.values(): h.remove()
    out['pred_a_content_causal_middle'] = bool(all(out['by_layer'][str(L)]['content_topic_net'] > out['by_layer'][str(L)]['random_topic_net'] + 0.05 for L in [8, 11]))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) content causally addressable in the middle (L8,L11 >> random): {out['pred_a_content_causal_middle']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
