"""BENCHMARK REFRESH: does a SMOOTH per-token LINEAR MAP (from the token embedding) beat the discrete token TABLE
as the token stand-in in the whole-model understanding benchmark? §936/§938 showed that, per component, a linear
map from the current-token embedding recovers MORE of the content than a discrete per-token table (it generalizes
across embedding-similar tokens; the table undersamples rare tokens). A linear map from the INPUT EMBEDDING is a
fair NAMED variable (a smooth function of token identity), NOT output leakage (unlike a high-rank subspace of the
component's own output, which §913 correctly rejected). So swap the token stand-in from table -> linear map and
re-measure the whole-model understanding fraction, held-out, against the table baseline (token+topic+prev ~0.315,
genuine vs shuffled ~0.43).

All 36 components replaced at once; tables/maps/subspaces fit on the first 70% of rows, CE evaluated on the
held-out 30%. Scale: 0 = all-mean-ablated, 1 = full model.

REGISTERED PREDICTIONS:
  (0) SANITY: all-mean CE >> full; table baseline reproduces ~0.31; shuffled-token null is low/negative.
  (a) LINEAR MAP HELPS: tokenmap+topic+prev recovers MORE than table token+topic+prev (a smooth per-token map is
      a better named-variable stand-in) -> the benchmark improves; but stays well below 1 (content continuum
      remains, §930/§938);
  (b) report understanding_frac for tokenmap / tokenmap+topic / tokenmap+topic+prev vs the table baseline and the
      shuffled null; the gain (map - table) is the smooth-generalization contribution."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_understanding_tokenmap_results.json'
NEVAL = 200; SEQ = 256; CONTENT_L = 15; K = 12; RTOK = 64; RPOS = 32; RIDGE_MAP = 1e3; MINCOUNT = 12
NLAYER = 18
REPL = {'mode': 'off', 'row': 0, 'standins': None, 'gmeans': None, 'level': None}
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
            yn = REPL['standins'][REPL['level']][tag][REPL['row']:REPL['row']+B*T].reshape(B, T, D).to(DEV)
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
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True
    toks = S[:, :-1].reshape(-1); prev = np.full((nb, SEQ-1), -1, dtype=np.int64); prev[:, 1:] = S[:, :-2]; prev = prev.reshape(-1)
    uniq = np.unique(np.concatenate([toks, prev[prev >= 0]])); remap = {int(t): j for j, t in enumerate(uniq)}; nu = len(uniq)
    import tiktoken; _enc = tiktoken.get_encoding('gpt2'); _dd = lambda i: _enc.decode([int(i)])
    _DET={'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
    _PREP={'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
    _CONJ={'and','or','but','nor','so','yet','because','although','while','if','than'}
    _PRON={'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}
    def _clsf(sx):
        t=sx.strip()
        if t=='' or not t[0].isalnum(): return 5
        if t[0].isdigit(): return 4
        low=t.lower()
        if low in _DET: return 0
        if low in _PREP: return 1
        if low in _CONJ: return 2
        if low in _PRON: return 3
        return 6 if t[0].isupper() else 7
    ucls = torch.tensor(np.array([_clsf(_dd(int(t))) for t in uniq]), device=DEV)
    tok_i = np.vectorize(lambda t: remap[int(t)])(toks).astype(np.int64)
    prev_i = np.array([remap[int(t)] if t >= 0 else 0 for t in prev], dtype=np.int64)
    posarr = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    # per-unique-token input embedding (x0 = rms_norm(wte(token)); purely per-token)
    Emb_u = F.rms_norm(m.transformer.wte(torch.tensor(uniq, device=DEV)), (D,)).float()  # (nu, D)
    Emb_all = Emb_u[torch.tensor(tok_i, device=DEV)]  # (N, D)
    # topic labels from L15 content
    cap15 = []
    def hc(mo, i_, o_): cap15.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = m.transformer.h[CONTENT_L].register_forward_hook(hc)
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    hh.remove(); R15 = torch.cat(cap15, 0)
    Utok, g = mean_subspace(R15, toks, RTOK); Upos, _ = mean_subspace(R15, posarr.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g) - ((R15-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic_i = kmeans(cn, K).cpu().numpy().astype(np.int64)
    rng = np.random.RandomState(0); tok_sh = rng.permutation(nu)[tok_i]
    # capture all 36 component outputs
    cap = {t: [] for t in TAGS}; caph = []
    for t in TAGS:
        def mk(t):
            def h(mo, i_, o_): cap[t].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        caph.append(submod(t).register_forward_hook(mk(t)))
    REPL['mode'] = 'off'
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in caph: h.remove()
    trmask = np.repeat(TRAIN, SEQ-1); trm_t = torch.tensor(trmask, device=DEV)
    Etr = Emb_all[trm_t]; AtA = Etr.T @ Etr + RIDGE_MAP*torch.eye(D, device=DEV)
    standins = {'token_table': {}, 'tokenmap': {}, 'tokenmap+topic': {}, 'tokenmap+topic+prev': {}, 'table+topic+prev': {}, 'shuftoken_map': {}}
    gmeans = {}
    tik = torch.tensor(tok_i, device=DEV); pik = torch.tensor(prev_i, device=DEV); tsh = torch.tensor(tok_sh, device=DEV)
    for t in TAGS:
        O = torch.cat(cap[t], 0); cap[t] = None
        # linear map token stand-in
        Mtok = torch.linalg.solve(AtA, Etr.T @ O[trm_t]); s_map = Emb_all @ Mtok
        # table token stand-in (with class-backoff) for baseline
        tt = table(O[trm_t], tok_i[trmask], nu)
        _cnt = torch.zeros(nu, device=DEV); _tik_tr = torch.tensor(tok_i[trmask], device=DEV); _cnt.index_add_(0, _tik_tr, torch.ones_like(_tik_tr, dtype=torch.float))
        _cm = torch.zeros(8, D, device=DEV); _cc = torch.zeros(8, device=DEV)
        _cm.index_add_(0, ucls[_tik_tr], O[trm_t]); _cc.index_add_(0, ucls[_tik_tr], torch.ones(int(trm_t.sum()), device=DEV)); _cm = _cm/_cc.clamp_min(1).unsqueeze(1)
        _rare = _cnt < MINCOUNT; tt[_rare] = _cm[ucls][_rare]
        # topic + prev built on the MAP residual
        r1 = O - s_map
        Utop, _ = mean_subspace(r1[trm_t], topic_i[trmask], K-1); top_term = (r1 @ Utop) @ Utop.T
        r2 = r1 - top_term; pv = table(r2[trm_t], prev_i[trmask], nu)
        # table baseline topic+prev built on TABLE residual
        r1b = O - tt[tik]; Utopb, _ = mean_subspace(r1b[trm_t], topic_i[trmask], K-1); top_b = (r1b @ Utopb) @ Utopb.T
        r2b = r1b - top_b; pvb = table(r2b[trm_t], prev_i[trmask], nu)
        standins['token_table'][t] = tt[tik].cpu()
        standins['tokenmap'][t] = s_map.cpu()
        standins['tokenmap+topic'][t] = (s_map + top_term).cpu()
        standins['tokenmap+topic+prev'][t] = (s_map + top_term + pv[pik]).cpu()
        standins['table+topic+prev'][t] = (tt[tik] + top_b + pvb[pik]).cpu()
        standins['shuftoken_map'][t] = (Emb_u[tsh] @ Mtok).cpu()
        gmeans[t] = O.mean(0); del O, r1, r2, r1b, r2b
    REPL['standins'] = standins; REPL['gmeans'] = gmeans
    test_blocks = blocks[~TRAIN]; ROW_OFFSET[0] = ntr*(SEQ-1)
    REPL['mode'] = 'off'; ce_full = ce_pass(test_blocks)
    REPL['mode'] = 'mean'; ce_mean = ce_pass(test_blocks)
    denom = max(ce_mean - ce_full, 1e-6); out = {'ce_full': round(ce_full, 3), 'ce_all_mean_ablated': round(ce_mean, 3), 'levels': {}}
    REPL['mode'] = 'set'
    for lv in ['token_table', 'tokenmap', 'tokenmap+topic', 'tokenmap+topic+prev', 'table+topic+prev', 'shuftoken_map']:
        REPL['level'] = lv; ce = ce_pass(test_blocks)
        out['levels'][lv] = {'ce': round(ce, 3), 'understanding_frac': round(float((ce_mean - ce)/denom), 3)}
        print(f"all-36 {lv:>22}: CE {ce:.3f} | understanding {out['levels'][lv]['understanding_frac']:.3f}", flush=True)
    mapf = out['levels']['tokenmap+topic+prev']['understanding_frac']; tabf = out['levels']['table+topic+prev']['understanding_frac']
    out['map_minus_table'] = round(mapf - tabf, 3)
    out['genuine_frac_vs_shuffled'] = round(mapf - out['levels']['shuftoken_map']['understanding_frac'], 3)
    out['pred_a_linear_map_helps'] = bool(mapf > tabf and out['genuine_frac_vs_shuffled'] > 0.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nMAP token+topic+prev {mapf} vs TABLE {tabf} (gain {out['map_minus_table']:+.3f}); genuine vs shuffled {out['genuine_frac_vs_shuffled']}", flush=True)
    print(f"(a) linear map helps the benchmark: {out['pred_a_linear_map_helps']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
