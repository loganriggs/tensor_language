"""[MULTI-DRAW stable estimate, §1014] §1013 found the whole-model understanding number is DRAW-SENSITIVE (0.29<->0.45). Run the fresh held-out benchmark over SEVERAL fineweb draws (offsets 0/150/300/450) and report the MEAN +- SPREAD of the understanding fraction, giving a stable headline instead of a draw-dependent point. REGISTERED: report mean+-std of token / token+topic+prev / genuine-vs-shuffled across draws; the token term should be tight, the topic-driven total wider (per §1013). [orig §900-fresh] CAPSTONE: HOW MUCH OF THE WHOLE MODEL DO WE UNDERSTAND? (user's core question). Tables/subspaces BUILT on the first 70% of rows, CE EVALUATED on the held-out 30% — certifies the 0.81 is not same-data optimism. Replace EVERY component's
output at once with its named-variable stand-in — token-table + continuous-topic-subspace + prev-token-table
(the variables we have named and causally verified: grammar/class §892, topic §894, induction/prev §877) —
and measure the full-model CE on the scale 0 = all-components-mean-ablated (know nothing) → 1 = full model.
This is the aggregate of the per-component metric (§893/896/898), all 36 components simultaneously, so errors
compound honestly: it is the fraction of the MODEL, not of one layer, that our named variables reconstruct.

Levels (all applied to all 36 components at once): all-mean (0), token-only, token+topic, token+topic+prev,
and a shuffled-token null (destroys the token map — controls the §836 rank artifact). Also report the FRONT-
only and per-band contributions.

REGISTERED PREDICTIONS:
  (0) SANITY: all-mean CE >> full CE; token-only recovers a large fraction (front is token-tables);
  (a) NAMED VARIABLES RECONSTRUCT MOST OF THE MODEL: token+topic+prev on all 36 components recovers a large,
      well-above-shuffled-null fraction of the model (target: majority) -> we understand most of the model as
      named variables, with a stated residual (long-range/interaction) we can name mechanistically but not
      tabulate;
  (b) report the aggregate fraction at each level + the shuffled null; the residual = the honest
      not-yet-tabulated remainder."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_understanding_multidraw_results.json'
NEVAL = 150; SEQ = 256; CONTENT_L = 15; K = 12; RTOK = 64; RPOS = 32
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
    standins = {'token': {}, 'token+topic': {}, 'token+topic+prev': {}, 'shuftoken': {}}; gmeans = {}
    tik = torch.tensor(tok_i, device=DEV); pik = torch.tensor(prev_i, device=DEV); tsh = torch.tensor(tok_sh, device=DEV)
    for t in TAGS:
        O = torch.cat(cap[t], 0); cap[t] = None
        trmask = np.repeat(TRAIN, SEQ-1)
        tt = table(O[trmask], tok_i[trmask], nu); r1 = O - tt[tik]
        Utop, _ = mean_subspace(r1[torch.tensor(trmask, device=DEV)], topic_i[trmask], K-1); top_term = (r1 @ Utop) @ Utop.T
        r2 = r1 - top_term; pv = table(r2[torch.tensor(trmask, device=DEV)], prev_i[trmask], nu)
        s_tok = tt[tik]; s_tt = s_tok + top_term; s_ttp = s_tt + pv[pik]; s_sh = tt[tsh]
        standins['token'][t] = s_tok.cpu(); standins['token+topic'][t] = s_tt.cpu()
        standins['token+topic+prev'][t] = s_ttp.cpu(); standins['shuftoken'][t] = s_sh.cpu()
        gmeans[t] = O.mean(0); del O, r1, r2
    REPL['standins'] = standins; REPL['gmeans'] = gmeans
    test_blocks = blocks[~TRAIN]; ROW_OFFSET[0] = ntr*(SEQ-1); REPL['mode'] = 'off'; ce_full = ce_pass(test_blocks)
    REPL['mode'] = 'mean'; ce_mean = ce_pass(test_blocks)
    denom = max(ce_mean - ce_full, 1e-6); out = {'ce_full': round(ce_full, 3), 'ce_all_mean_ablated': round(ce_mean, 3), 'levels': {}}
    REPL['mode'] = 'set'
    for lv in ['token', 'token+topic', 'token+topic+prev', 'shuftoken']:
        REPL['level'] = lv; ce = ce_pass(test_blocks)
        out['levels'][lv] = {'ce': round(ce, 3), 'understanding_frac': round(float((ce_mean - ce)/denom), 3)}
        print(f"all-36 {lv:>16}: CE {ce:.3f} | understanding {out['levels'][lv]['understanding_frac']:.3f}", flush=True)
    for h in hooks: h.remove()
    g_frac = out['levels']['token+topic+prev']['understanding_frac'] - out['levels']['shuftoken']['understanding_frac']
    out['genuine_frac_vs_shuffled'] = round(g_frac, 3)
    out['pred_a_named_vars_reconstruct_most'] = bool(out['levels']['token+topic+prev']['understanding_frac'] > 0.5 and g_frac > 0.3)
    out['runtime_s'] = round(time.time()-t0, 1)
    return out


@torch.no_grad()
def main():
    cl.use_state(PT + 'census_state_diverse.pt')
    OFFSETS = [0, 150, 300, 450]
    draws = []
    for off in OFFSETS:
        rows = cl.fineweb_rows(off + NEVAL)[off:]
        r = run_once(rows)
        draws.append({'offset': off, 'token': r['levels']['token']['understanding_frac'],
                      'token+topic+prev': r['levels']['token+topic+prev']['understanding_frac'],
                      'genuine_vs_shuffled': r['genuine_frac_vs_shuffled'], 'ce_full': r['ce_full']})
        print(f"draw off={off}: token {draws[-1]['token']} ttp {draws[-1]['token+topic+prev']} genuine {draws[-1]['genuine_vs_shuffled']}", flush=True)
    import statistics as st
    def ms(key):
        vals = [d[key] for d in draws]; return {'mean': round(st.mean(vals), 3), 'std': round(st.pstdev(vals), 3), 'min': round(min(vals), 3), 'max': round(max(vals), 3)}
    out = {'draws': draws, 'token': ms('token'), 'token+topic+prev': ms('token+topic+prev'),
           'genuine_vs_shuffled': ms('genuine_vs_shuffled')}
    out['pred_token_tight'] = bool(out['token']['std'] < 0.05)
    out['pred_total_wider'] = bool(out['token+topic+prev']['std'] > out['token']['std'])
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"MULTI-DRAW: token {out['token']} | token+topic+prev {out['token+topic+prev']} | genuine {out['genuine_vs_shuffled']}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == '__main__':
    main()
