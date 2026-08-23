"""PER-COMPONENT understanding graph (§1035): for EACH of the 36 components (attn/mlp x 18 layers), how much of its
INDIVIDUAL contribution do our named variables (token-table + continuous-topic-subspace + prev-token-table) recover?
For each component in isolation: frac = (CE[mean-ablate only this] - CE[named-standin only this]) / (CE[mean-ablate
only this] - CE_full). This gives the benchmark broken down component-by-component (the "graph across all module
components"), held-out (stand-ins built on the first 70% of rows, evaluated on the last 30%).

REGISTERED PREDICTIONS:
  (0) SANITY: shuffled-token stand-in recovers ~0 per component; full model CE reproduced with no replacement.
  (a) TWO-MACHINE DEPTH STRUCTURE AT COMPONENT GRANULARITY: FRONT components (esp mlp0-2, attn0-2) have HIGH
      per-component understanding (grammar/token machine, ~0.6-0.9); MIDDLE components (L6-11) have LOW understanding
      (content machine, the multiplicative frontier, ~0.1-0.3); readout (L16-17) moderate-high (linear readout);
  (b) report the per-component fraction for all 36 + means by band and by type (attn vs mlp)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'per_component_understanding_results.json'
NEVAL = 200; SEQ = 256; CONTENT_L = 15; K = 12; RTOK = 64; RPOS = 32
NLAYER = 18
REPL = {'mode': 'off', 'row': 0, 'standins': None, 'gmeans': None, 'level': None, 'only': None}
ROW_OFFSET = [0]
TAGS = [f"{k}{L}" for L in range(NLAYER) for k in ('attn', 'mlp')]


def submod(tag):
    k = 'attn' if tag.startswith('attn') else 'mlp'; L = int(tag[len(k):]); return getattr(m.transformer.h[L], k)


def repl_hook_factory(tag):
    def hook(mo, i_, o_):
        if REPL['mode'] == 'off': return o_
        if REPL.get('only') is not None and REPL['only'] != tag: return o_
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
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
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
    out = {'ce_full': round(ce_full, 3), 'per_component': {}, 'per_component_shuf': {}}
    for tag in TAGS:
        REPL['only'] = tag
        REPL['mode'] = 'mean'; ce_ma = ce_pass(test_blocks)
        denom = max(ce_ma - ce_full, 1e-6)
        REPL['mode'] = 'set'; REPL['level'] = 'token+topic+prev'; ce_si = ce_pass(test_blocks)
        REPL['level'] = 'shuftoken'; ce_sh = ce_pass(test_blocks)
        out['per_component'][tag] = round(float((ce_ma - ce_si)/denom), 3)
        out['per_component_shuf'][tag] = round(float((ce_ma - ce_sh)/denom), 3)
        out['%s_meanabl_cost' % tag] = round(float(ce_ma - ce_full), 3)
        print(f"{tag:>7}: understanding {out['per_component'][tag]:+.3f} (shuf-null {out['per_component_shuf'][tag]:+.3f}, meanabl-cost {ce_ma-ce_full:.3f})", flush=True)
    REPL['only'] = None
    for h in hooks: h.remove()
    pc = out['per_component']
    def band(ls): 
        v=[pc[f'{k}{L}'] for L in ls for k in ('attn','mlp')]; return round(float(np.mean(v)),3)
    out['band_front_L0_5'] = band(range(0,6)); out['band_middle_L6_11'] = band(range(6,12)); out['band_back_L12_17'] = band(range(12,18))
    out['mean_attn'] = round(float(np.mean([pc[f'attn{L}'] for L in range(NLAYER)])),3)
    out['mean_mlp'] = round(float(np.mean([pc[f'mlp{L}'] for L in range(NLAYER)])),3)
    out['pred_a_front_gt_middle'] = bool(out['band_front_L0_5'] > out['band_middle_L6_11'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nBANDS: front {out['band_front_L0_5']} | middle {out['band_middle_L6_11']} | back {out['band_back_L12_17']} | attn {out['mean_attn']} mlp {out['mean_mlp']}", flush=True)
    print(f"(a) front > middle per-component: {out['pred_a_front_gt_middle']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
