"""[content-PCA-128: higher-rank continuous content feature (top-128 PCA of L15 content residual), fixing the §907 cluster-cap no-op] GAP-FILL: is the middle content a SMOOTH (generalizing) function of named variables, or genuinely
irreducible? Tables MEMORIZE (§901-903: middle token-tables don't generalize). But a discrete lookup fails on
unseen token/topic combos even if the true mapping is smooth. Fit instead a LOW-RANK LINEAR MAP from the named
variables — current-token embedding, previous-token embedding, and the continuous-topic-subspace projection —
to each component's output, on TRAIN rows, and test on held-out rows. If the smooth map generalizes far better
than the table for the middle, the middle content IS an understandable (smooth, low-parameter) function of
named variables; if it too fails, the middle is genuinely high-dimensional/irreducible beyond these variables.

REGISTERED PREDICTIONS:
  (0) SANITY: for mlp0 the map ≈ the token table (front is token-determined); shuffled-feature null ~0;
  (a) MIDDLE IS A SMOOTH FUNCTION: the learned map's fresh understanding for the middle content components
      (mlp5/mlp8/mlp11/attn5) is clearly ABOVE their table's fresh ~0 (§903) and above the shuffled-feature
      null -> the middle is a smooth generalizing function of token+prev+topic (understandable, not a lookup);
  (b) if the map is also ~0 fresh, the middle is genuinely irreducible beyond these named variables (report
      plainly — the honest ceiling of this variable set)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'learned_map_contentpca_results.json'
NEVAL = 200; SEQ = 256; CONTENT_L = 15; K = 12; RTOK = 64; RPOS = 32; RPCA = 128; RIDGE = 1e2
COMPS = [(0, 'mlp'), (5, 'attn'), (5, 'mlp'), (8, 'mlp'), (11, 'mlp'), (16, 'mlp')]
SAMEDATA = {'mlp0': 0.94, 'attn5': 0.50, 'mlp5': 0.39, 'mlp8': 0.47, 'mlp11': 0.48, 'mlp16': 0.78}
TABLE_FRESH = {'mlp0': 0.59, 'attn5': 0.02, 'mlp5': 0.02, 'mlp8': -0.09, 'mlp11': -0.06, 'mlp16': 0.37}
REPL = {'mode': 'off', 'target': None, 'val': None, 'gmean': None}


def submod(L, kind): return getattr(m.transformer.h[L], kind)


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off' or REPL['target'] != tag: return o_
        y = o_[0] if isinstance(o_, tuple) else o_; B, T, _ = y.shape
        yn = REPL['gmean'].expand(B, T, D).clone() if REPL['mode'] == 'mean' else REPL['val']
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


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    ntr = int(0.7*nb); TRAIN = np.zeros(nb, bool); TRAIN[:ntr] = True; trpos = np.repeat(TRAIN, SEQ-1)
    wte = m.transformer.wte.weight.detach().float()
    cur = S[:, :-1].reshape(-1); prev = np.full((nb, SEQ-1), -1, dtype=np.int64); prev[:, 1:] = S[:, :-2]; prev = prev.reshape(-1)
    # topic subspace projection per position (from L15 content)
    tags = [f"{k}{L}" for (L, k) in COMPS]; hooks = [submod(L, k).register_forward_hook(repl_hook_factory(f"{k}{L}")) for (L, k) in COMPS]
    cap = {}
    caph = []
    for (L, k) in COMPS:
        tag = f"{k}{L}"
        def mk(tag):
            def h(mo, i_, o_): cap[tag] = (o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D)
            return h
        caph.append(submod(L, k).register_forward_hook(mk(tag)))
    c15 = []
    def hc(mo, i_, o_): c15.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hc_h = m.transformer.h[CONTENT_L].register_forward_hook(hc)
    REPL['mode'] = 'off'; caps_all = {t: [] for t in tags}
    for i in range(0, nb, 8):
        forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
        for t in tags: caps_all[t].append(cap[t])
    for h in caph: h.remove(); hc_h.remove()
    outs = {t: torch.cat(caps_all[t], 0) for t in tags}
    R15 = torch.cat(c15, 0)
    toks = cur; posarr = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    Utok, g = mean_subspace(R15, toks, RTOK); Upos, _ = mean_subspace(R15, posarr.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g) - ((R15-g)@Ucp)@Ucp.T
    _cc = content - content.mean(0, keepdim=True)
    Vpca = torch.linalg.svd(_cc, full_matrices=False)[2][:RPCA].T.contiguous()  # (D, 128) top content PCA
    topic_proj = content @ Vpca                                     # (N, 128) higher-rank continuous content
    # feature matrix: [wte[cur], wte[prev], topic_proj, 1]
    feat = torch.cat([wte[torch.tensor(cur, device=DEV)], wte[torch.tensor(np.where(prev>=0,prev,0), device=DEV)],
                      topic_proj, torch.ones(len(cur), 1, device=DEV)], 1)
    rng = np.random.RandomState(0); featperm = feat[torch.tensor(rng.permutation(len(cur)), device=DEV)]  # shuffled-feature null
    tr = torch.tensor(trpos, device=DEV); te = ~tr
    Ftr = feat[tr]; A = Ftr.T @ Ftr + RIDGE*torch.eye(feat.shape[1], device=DEV)
    Ftr_sh = featperm[tr]; A_sh = Ftr_sh.T @ Ftr_sh + RIDGE*torch.eye(feat.shape[1], device=DEV)
    # replace-forward helper on TEST blocks; standin tensor set per component
    te_blocks = blocks[~TRAIN]; te_tok = None
    def ce_test(valtensor_full):
        # valtensor_full: (N, D) prediction for ALL positions; slice test rows
        tot = 0.0; n = 0; row = ntr*(SEQ-1)
        for i in range(0, te_blocks.shape[0], 8):
            bb = te_blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous(); bsz = idx.shape[0]
            REPL['val'] = valtensor_full[row:row+bsz*(SEQ-1)].reshape(bsz, SEQ-1, D).to(DEV)
            lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel(); row += bsz*(SEQ-1)
        return tot/n
    out = {'components': {}}
    for (L, k) in COMPS:
        tag = f"{k}{L}"; O = outs[tag]; REPL['target'] = tag; REPL['gmean'] = O.mean(0)
        W = torch.linalg.solve(A, Ftr.T @ O[tr]); pred = feat @ W
        Wsh = torch.linalg.solve(A_sh, Ftr_sh.T @ O[tr]); pred_sh = featperm @ Wsh
        REPL['mode'] = 'mean'; ce_mean = ce_test(O)  # O unused in mean mode
        REPL['mode'] = 'set'; ce_map = ce_test(pred); ce_sh = ce_test(pred_sh)
        REPL['mode'] = 'off'
        denom = max(ce_mean - out.get('_full', None) if False else 1, 1e-6)  # placeholder; compute full below
        out['components'][tag] = {'_ce_mean': round(ce_mean,3), '_ce_map': round(ce_map,3), '_ce_sh': round(ce_sh,3)}
    # full CE (no replacement) on test
    REPL['mode'] = 'off'; row = ntr*(SEQ-1); tot=0.0; n=0
    for i in range(0, te_blocks.shape[0], 8):
        bb = te_blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1,lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n+=tgt.numel()
    ce_full = tot/n; out['ce_full'] = round(ce_full,3)
    for tag in [f"{k}{L}" for (L,k) in COMPS]:
        c = out['components'][tag]; denom = max(c['_ce_mean']-ce_full, 1e-6)
        c['map_fresh'] = round((c['_ce_mean']-c['_ce_map'])/denom, 3); c['shuffled_null'] = round((c['_ce_mean']-c['_ce_sh'])/denom, 3)
        c['table_fresh_ref'] = TABLE_FRESH.get(tag); c['samedata_ref'] = SAMEDATA.get(tag)
        print(f"{tag:>6}: MAP fresh {c['map_fresh']:.2f} (null {c['shuffled_null']:+.2f}) | table fresh {TABLE_FRESH.get(tag)} | same-data {SAMEDATA.get(tag)}", flush=True)
    for h in hooks: h.remove()
    mids = ['attn5','mlp5','mlp8','mlp11']
    out['mean_map_fresh_middle'] = round(float(np.mean([out['components'][t]['map_fresh'] for t in mids])),3)
    out['mean_table_fresh_middle'] = round(float(np.mean([TABLE_FRESH[t] for t in mids])),3)
    out['pred_a_middle_smooth'] = bool(out['mean_map_fresh_middle'] > out['mean_table_fresh_middle'] + 0.1 and out['mean_map_fresh_middle'] > 0.15)
    out['runtime_s'] = round(time.time()-t0,1)
    json.dump(out, open(OUT,'w'), indent=1)
    print(f"\nmiddle: MAP fresh {out['mean_map_fresh_middle']} vs TABLE fresh {out['mean_table_fresh_middle']}", flush=True)
    print(f"(a) middle is a smooth generalizing function of named variables: {out['pred_a_middle_smooth']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
