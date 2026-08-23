"""NORTH-STAR REFRESH (registered §1120): the §939/§1013 benchmark's content term is 12 coarse k-means topic
buckets; the skeleton arc (§1113-1119) found the content READ API = 8 stable features. Upgrade the content term:
per component, the residual-after-token-table is fit by a LINEAR MAP from the 8 skeleton activations (train
rows only; skeleton = k=8/256 SAE on pooled L8-12 content coords, trained on train rows). Levels: token |
token+topic(orig, 12 buckets) | token+skel | token+skel+prev | shuffled-skel null (activations permuted across
positions). 2 draws.

REGISTERED PREDICTIONS:
  (0) SANITY: token level reproduces ~0.15-0.32-range prior numbers; shuffled-skel adds ~0 over token.
  (a) SKELETON BEATS TOPIC: token+skel > token+topic (8 named features beat 12 buckets) and token+skel+prev
      lifts the held-out understanding band above the standing 0.32±0.06;
  (b) SCRATCH-LIMITED (the §1118 expectation): if token+skel ≈ token+topic (within 0.03), the benchmark's
      content gap is NOT a read-interface problem — per-module output substitution has to reproduce the
      CONSTRUCTION (whose scratch no small feature set carries, §1118), so even the true API doesn't lift
      tabulation. That result would quantify, at the north-star level, that the remaining gap is construction
      simulation — report plainly either way."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'benchmark_skeleton_results.json'
NEVAL = 150; SEQ = 256; CONTENT_L = 15; K = 12; RTOK = 64; RPOS = 32
NLAYER = 18
REPL = {'mode': 'off', 'row': 0, 'standins': None, 'gmeans': None, 'level': None}
REFC = [8, 10, 12]; KC = 64; NATOM = 256; TOPK = 8; SAESTEPS = 2500


class TopKSAE(torch.nn.Module):
    def __init__(self, d, n, k, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.E = torch.nn.Parameter(torch.randn(n, d, generator=g)*0.1)
        self.Dm = torch.nn.Parameter(torch.randn(d, n, generator=g)*0.1)
        self.k = k
    def forward(self, x):
        a = x @ self.E.T
        top = a.topk(self.k, -1)
        code = torch.zeros_like(a).scatter_(-1, top.indices, top.values)
        return code @ self.Dm.T, code
ROW_OFFSET = [0]
TAGS = [f"{k}{L}" for L in range(NLAYER) for k in ('attn', 'mlp')]


def submod(tag):
    k = 'attn' if tag.startswith('attn') else 'mlp'; L = int(tag[len(k):]); return getattr(m.transformer.h[L], k)


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off': return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        if REPL['mode'] == 'mean':
            yn = REPL['gmeans'][tag].expand(B, T, D).clone()
        else:
            sl = REPL['standins'][REPL['level']][tag][REPL['row']:REPL['row']+B*T].reshape(B, T, D).to(DEV)
            yn = sl
        return (yn,) + tuple(o_[1:]) if isinstance(o_, tuple) else yn
    return hook


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def kmeans(X, k, iters=25, seed=0):
    g = torch.Generator(device=X.device).manual_seed(seed); c = X[torch.randperm(X.shape[0], generator=g, device=X.device)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(X, c).argmin(1)
        for j in range(k):
            mk = a == j
            if mk.any(): c[j] = X[mk].mean(0)
    return a


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []; wt = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[mk].mean(0)-g[0]); wt.append(np.sqrt(mk.sum()))
    M = torch.stack(rows, 0)*torch.tensor(wt, device=X.device, dtype=X.dtype)[:, None]
    return torch.linalg.svd(M, full_matrices=False)[2][:min(r, M.shape[0])].T.contiguous(), g


def table(resid, key, ncat):
    t = torch.zeros(ncat, D, device=DEV); c = torch.zeros(ncat, device=DEV); k = torch.tensor(key, device=DEV)
    t.index_add_(0, k, resid); c.index_add_(0, k, torch.ones_like(k, dtype=torch.float)); return t / c.clamp_min(1).unsqueeze(1)


@torch.no_grad()
def ce_pass(blocks):
    tot = 0.0; n = 0; REPL['row'] = ROW_OFFSET[0]
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum'))
        n += tgt.numel(); REPL['row'] += idx.shape[0]*(SEQ-1)
    return tot / n


@torch.no_grad()
def run_once(rows):
    t0 = time.time()
    REPL['mode'] = 'off'
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True  # first 70% seqs build tables; last 30% eval
    toks = S[:, :-1].reshape(-1); prev = np.full((nb, SEQ-1), -1, dtype=np.int64); prev[:, 1:] = S[:, :-2]; prev = prev.reshape(-1)
    uniq = np.unique(np.concatenate([toks, prev[prev >= 0]])); remap = {int(t): j for j, t in enumerate(uniq)}; nu = len(uniq)
    tok_i = np.vectorize(lambda t: remap[int(t)])(toks).astype(np.int64)
    prev_i = np.array([remap[int(t)] if t >= 0 else 0 for t in prev], dtype=np.int64)
    posarr = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    # topic labels from L15 content
    cap15 = []
    def hc(mo, i_, o_): cap15.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = m.transformer.h[CONTENT_L].register_forward_hook(hc)
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    hh.remove()
    R15 = torch.cat(cap15, 0)
    Utok, g = mean_subspace(R15, toks, RTOK); Upos, _ = mean_subspace(R15, posarr.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g) - ((R15-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic_i = kmeans(cn, K).cpu().numpy().astype(np.int64)
    rng = np.random.RandomState(0); tok_sh = rng.permutation(nu)[tok_i]
    # skeleton activations: pooled L8-12 mlp-input content coords -> SAE(k=8) code, active-atom top-8 features
    capC = {L: [] for L in REFC}; hsC = []
    for L in REFC:
        def mkc(L):
            def h(mo, i_, o_): capC[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hsC.append(m.transformer.h[L].mlp.register_forward_hook(mkc(L)))
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hsC: h.remove()
    tokt = torch.tensor(tok_i, device=DEV); V2 = nu
    devsum = None
    for L in REFC:
        X = torch.cat(capC[L], 0); capC[L] = []
        xb = torch.zeros(V2, D, device=DEV); cnn = torch.zeros(V2, device=DEV)
        xb.index_add_(0, tokt, X); cnn.index_add_(0, tokt, torch.ones_like(tokt, dtype=torch.float))
        dvv = X - (xb/cnn.clamp_min(1).unsqueeze(1))[tokt]
        devsum = dvv if devsum is None else devsum + dvv; del X
    devc2 = devsum/len(REFC); devc2 = devc2 - devc2.mean(0)
    _, _, Vt2 = torch.linalg.svd(devc2, full_matrices=False)
    Cc2 = (devc2 @ Vt2[:KC].T.contiguous()).contiguous(); del devc2, devsum
    trmask_flat = torch.tensor(np.repeat(TRAIN, SEQ-1), device=DEV)
    sae = TopKSAE(KC, NATOM, TOPK, 0).to(DEV)
    optS = torch.optim.Adam(sae.parameters(), lr=3e-3)
    Ctr = Cc2[trmask_flat]
    with torch.enable_grad():
        for step in range(SAESTEPS):
            ii = torch.randint(0, Ctr.shape[0], (4096,), device=DEV)
            xh2, _ = sae(Ctr[ii]); lossS = ((xh2 - Ctr[ii])**2).mean()
            optS.zero_grad(); lossS.backward(); optS.step()
    with torch.no_grad():
        _, codeAll = sae(Cc2)
    usage = (codeAll[trmask_flat] != 0).float().mean(0)
    top8a = usage.argsort(descending=True)[:8]
    SKEL = codeAll[:, top8a]                        # N x 8 skeleton activations
    permN = torch.randperm(SKEL.shape[0], generator=torch.Generator(device=DEV).manual_seed(3), device=DEV)
    SKEL_SH = SKEL[permN]
    del Cc2, codeAll
    # capture all 36 component outputs
    hooks = [submod(t).register_forward_hook(repl_hook_factory(t)) for t in TAGS]
    cap = {t: [] for t in TAGS}; caph = []
    for t in TAGS:
        def mk(t):
            def h(mo, i_, o_): cap[t].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        caph.append(submod(t).register_forward_hook(mk(t)))
    REPL['mode'] = 'off'
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in caph: h.remove()
    # build stand-ins per component (on CPU to save GPU mem)
    standins = {'token': {}, 'token+topic': {}, 'token+skel': {}, 'token+skel+prev': {}, 'token+topic+prev': {}, 'shufskel': {}}; gmeans = {}
    tik = torch.tensor(tok_i, device=DEV); pik = torch.tensor(prev_i, device=DEV); tsh = torch.tensor(tok_sh, device=DEV)
    for t in TAGS:
        O = torch.cat(cap[t], 0); cap[t] = None
        trmask = np.repeat(TRAIN, SEQ-1)
        tt = table(O[trmask], tok_i[trmask], nu); r1 = O - tt[tik]
        trm = torch.tensor(trmask, device=DEV)
        Utop, _ = mean_subspace(r1[trm], topic_i[trmask], K-1); top_term = (r1 @ Utop) @ Utop.T
        # skeleton term: linear map 8->D fit on train rows (least squares with intercept)
        A1 = torch.cat([SKEL[trm], torch.ones(int(trm.sum()), 1, device=DEV)], 1)
        W8 = torch.linalg.lstsq(A1, r1[trm]).solution                  # 9 x D
        skel_term = torch.cat([SKEL, torch.ones(SKEL.shape[0], 1, device=DEV)], 1) @ W8
        skel_term_sh = torch.cat([SKEL_SH, torch.ones(SKEL.shape[0], 1, device=DEV)], 1) @ W8
        r2 = r1 - top_term; pv = table(r2[trm], prev_i[trmask], nu)
        r2s = r1 - skel_term; pvs = table(r2s[trm], prev_i[trmask], nu)
        s_tok = tt[tik]
        standins['token'][t] = s_tok.cpu()
        standins['token+topic'][t] = (s_tok + top_term).cpu()
        standins['token+skel'][t] = (s_tok + skel_term).cpu()
        standins['token+skel+prev'][t] = (s_tok + skel_term + pvs[pik]).cpu()
        standins['token+topic+prev'][t] = (s_tok + top_term + pv[pik]).cpu()
        standins['shufskel'][t] = (s_tok + skel_term_sh).cpu()
        gmeans[t] = O.mean(0); del O, r1, r2, r2s
    REPL['standins'] = standins; REPL['gmeans'] = gmeans
    test_blocks = blocks[~TRAIN]; ROW_OFFSET[0] = ntr*(SEQ-1); REPL['mode'] = 'off'; ce_full = ce_pass(test_blocks)
    REPL['mode'] = 'mean'; ce_mean = ce_pass(test_blocks)
    denom = max(ce_mean - ce_full, 1e-6); out = {'ce_full': round(ce_full, 3), 'ce_all_mean_ablated': round(ce_mean, 3), 'levels': {}}
    REPL['mode'] = 'set'
    for lv in ['token', 'token+topic', 'token+skel', 'token+skel+prev', 'token+topic+prev', 'shufskel']:
        REPL['level'] = lv; ce = ce_pass(test_blocks)
        out['levels'][lv] = {'ce': round(ce, 3), 'understanding_frac': round(float((ce_mean - ce)/denom), 3)}
        print(f"all-36 {lv:>16}: CE {ce:.3f} | understanding {out['levels'][lv]['understanding_frac']:.3f}", flush=True)
    for h in hooks: h.remove()
    out['skel_minus_topic'] = round(out['levels']['token+skel']['understanding_frac'] - out['levels']['token+topic']['understanding_frac'], 3)
    out['skel_null_gain'] = round(out['levels']['shufskel']['understanding_frac'] - out['levels']['token']['understanding_frac'], 3)
    out['runtime_s'] = round(time.time()-t0, 1)
    return out


@torch.no_grad()
def main():
    cl.use_state(PT + 'census_state_diverse.pt')
    OFFSETS = [0, 150]
    draws = []
    for off in OFFSETS:
        rows = cl.fineweb_rows(off + NEVAL)[off:]
        r = run_once(rows)
        draws.append({'offset': off, 'token': r['levels']['token']['understanding_frac'],
                      'token+topic': r['levels']['token+topic']['understanding_frac'],
                      'token+skel': r['levels']['token+skel']['understanding_frac'],
                      'token+skel+prev': r['levels']['token+skel+prev']['understanding_frac'],
                      'token+topic+prev': r['levels']['token+topic+prev']['understanding_frac'],
                      'shufskel': r['levels']['shufskel']['understanding_frac'],
                      'skel_minus_topic': r['skel_minus_topic'], 'ce_full': r['ce_full']})
        print(f"draw off={off}: {draws[-1]}", flush=True)
    import statistics as st
    def ms(key):
        vals = [d[key] for d in draws]; return {'mean': round(st.mean(vals), 3), 'std': round(st.pstdev(vals), 3), 'min': round(min(vals), 3), 'max': round(max(vals), 3)}
    out = {'draws': draws, 'token': ms('token'), 'token+skel': ms('token+skel'),
           'token+topic': ms('token+topic'), 'token+skel+prev': ms('token+skel+prev'),
           'token+topic+prev': ms('token+topic+prev'), 'skel_minus_topic': ms('skel_minus_topic')}
    out['pred_a_skeleton_beats_topic'] = bool(out['skel_minus_topic']['mean'] > 0.03)
    out['pred_b_scratch_limited'] = bool(abs(out['skel_minus_topic']['mean']) <= 0.03)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"MULTI-DRAW: token {out['token']} | skel {out['token+skel']} | topic {out['token+topic']} | skel-minus-topic {out['skel_minus_topic']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == '__main__':
    main()
