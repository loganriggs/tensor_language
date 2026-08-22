"""WHERE does the whole-model understanding (0.422, §939) come from, and WHERE is the residual? Localize the
benchmark across DEPTH. Using the best stand-in (per-token LINEAR MAP + topic + prev, §939), for each depth BAND
(front L0-5, middle L6-11, back L12-17) measure how much of THAT band's causal contribution the stand-in
recovers: replace only the band's components (rest of the model REAL), vs mean-ablating only the band (rest
REAL). understanding_band = (CE_band_meanablated - CE_band_standin)/(CE_band_meanablated - CE_full). This says
which layers we reconstruct as named variables and which are the residual.

REGISTERED PREDICTIONS:
  (0) SANITY: mean-ablating each band raises CE above full; shuffled-map per band is low.
  (a) FRONT UNDERSTOOD, BACK IS RESIDUAL: the FRONT band is reconstructed best by the named-variable stand-in
      (front = token tables + grammar, §915), the BACK band worst (back = content readout / high-rank continuum,
      §930/§938) -> understanding_front > understanding_back;
  (b) report per-band understanding_frac + the shuffled-map null per band."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'whole_model_understanding_bands_results.json'
NEVAL = 200; SEQ = 256; CONTENT_L = 15; K = 12; RTOK = 64; RPOS = 32; RIDGE_MAP = 1e3; MINCOUNT = 12
NLAYER = 18
BANDS = {'front_L0_5': list(range(0, 6)), 'middle_L6_11': list(range(6, 12)), 'back_L12_17': list(range(12, 18))}
REPL = {'mode': 'off', 'row': 0, 'standins': None, 'gmeans': None, 'level': None, 'active': set()}
ROW_OFFSET = [0]
TAGS = [f"{k}{L}" for L in range(NLAYER) for k in ('attn', 'mlp')]


def submod(tag):
    k = 'attn' if tag.startswith('attn') else 'mlp'; L = int(tag[len(k):]); return getattr(m.transformer.h[L], k)


def tag_layer(tag): return int(tag[4:]) if tag.startswith('attn') else int(tag[3:])


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off' or tag not in REPL['active']: return o_
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
    Emb_u = F.rms_norm(m.transformer.wte(torch.tensor(uniq, device=DEV)), (D,)).float()
    Emb_all = Emb_u[torch.tensor(tok_i, device=DEV)]
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
    standins = {'map+topic+prev': {}, 'shuftoken_map': {}}; gmeans = {}
    tik = torch.tensor(tok_i, device=DEV); pik = torch.tensor(prev_i, device=DEV); tsh = torch.tensor(tok_sh, device=DEV)
    for t in TAGS:
        O = torch.cat(cap[t], 0); cap[t] = None
        Mtok = torch.linalg.solve(AtA, Etr.T @ O[trm_t]); s_map = Emb_all @ Mtok
        r1 = O - s_map; Utop, _ = mean_subspace(r1[trm_t], topic_i[trmask], K-1); top_term = (r1 @ Utop) @ Utop.T
        r2 = r1 - top_term; pv = table(r2[trm_t], prev_i[trmask], nu)
        standins['map+topic+prev'][t] = (s_map + top_term + pv[pik]).cpu()
        standins['shuftoken_map'][t] = (Emb_u[tsh] @ Mtok).cpu()
        gmeans[t] = O.mean(0); del O, r1, r2
    REPL['standins'] = standins; REPL['gmeans'] = gmeans
    hooks = [submod(t).register_forward_hook(repl_hook_factory(t)) for t in TAGS]
    test_blocks = blocks[~TRAIN]; ROW_OFFSET[0] = ntr*(SEQ-1)
    REPL['mode'] = 'off'; REPL['active'] = set(); ce_full = ce_pass(test_blocks)
    out = {'ce_full': round(ce_full, 3), 'bands': {}}
    for bname, layers in BANDS.items():
        active = set(t for t in TAGS if tag_layer(t) in layers); REPL['active'] = active
        REPL['mode'] = 'mean'; ce_m = ce_pass(test_blocks)
        REPL['mode'] = 'set'; REPL['level'] = 'map+topic+prev'; ce_s = ce_pass(test_blocks)
        REPL['level'] = 'shuftoken_map'; ce_sh = ce_pass(test_blocks)
        denom = max(ce_m - ce_full, 1e-6)
        u = float((ce_m - ce_s)/denom); ush = float((ce_m - ce_sh)/denom)
        out['bands'][bname] = {'ce_meanablate': round(ce_m, 3), 'ce_standin': round(ce_s, 3),
                               'understanding_frac': round(u, 3), 'shuffled_frac': round(ush, 3)}
        print(f"{bname:>14}: meanabl {ce_m:.3f} standin {ce_s:.3f} -> understanding {u:.3f} (shuffled {ush:.3f})", flush=True)
    for h in hooks: h.remove()
    b = out['bands']
    out['pred_a_front_gt_back'] = bool(b['front_L0_5']['understanding_frac'] > b['back_L12_17']['understanding_frac'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"(a) front understood > back residual: {out['pred_a_front_gt_back']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
