"""FILL THE UNDERSTANDING GAPS with our named variables (user directive). §893: middle content MLPs are only
~0.4 understood by a token→output table (the context-free part). The GAP is context-dependent. Fill it with the
variables we have named — TOPIC (the gist, §866-894) and PREV-token (induction/local, §877) — built as
additive conditional-mean tables, and measure how much each closes the gap. This turns "we understand it as a
mechanism (gist) but not as a table" into a quantified table: token → token+topic → token+topic+prev.

Additive stand-in for a component's output:
  standin = table_tok[tok] + table_topic[topic] + table_prev[prev]
  (each table = mean of the residual left after the previous tables; topic from the §866 content clustering.)
Scale: 0 = mean-ablate, 1 = full model. Null: SHUFFLED topic labels for the topic table (controls the rank/
frequency artifact) — genuine topic fill = (token+topic) − (token+shuffled-topic).

REGISTERED PREDICTIONS:
  (0) SANITY: token-only reproduces §893 (mlp0 ~0.9, mlp8 ~0.4);
  (a) TOPIC FILLS THE MIDDLE GAP: for middle content MLPs (mlp5/8/11), token+topic >> token-only and >> the
      shuffled-topic null -> the gap is the topic/gist variable we understand; adding prev-token fills more of
      the induction-flavored part;
  (b) for FRONT components, topic adds little over token (they are context-free) — the increment is
      middle-specific. Report the understanding fraction at each level per component."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'layer_understanding_v2_results.json'
NEVAL = 160; SEQ = 256; CONTENT_L = 15; K = 12; RTOK = 64; RPOS = 32
COMPS = [(0, 'mlp'), (0, 'attn'), (5, 'attn'), (5, 'mlp'), (8, 'mlp'), (11, 'mlp'), (11, 'attn'), (16, 'mlp')]
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


def table(resid, key, ncat):
    """mean residual per category key (int array), returns (ncat, D)."""
    t = torch.zeros(ncat, D, device=DEV); c = torch.zeros(ncat, device=DEV)
    k = torch.tensor(key, device=DEV)
    t.index_add_(0, k, resid); c.index_add_(0, k, torch.ones_like(k, dtype=torch.float))
    return t / c.clamp_min(1).unsqueeze(1)


@torch.no_grad()
def ce_under(blocks, valid_from=1):
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        REPL['batch_i'] = i
        lg = forward_logits(idx).float()
        tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel()
    return tot / n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    idxcols = SEQ - 1
    toks = S[:, :-1].reshape(-1)
    prev = np.full((nb, SEQ-1), -1, dtype=np.int64); prev[:, 1:] = S[:, :-2]; prev = prev.reshape(-1)
    uniq = np.unique(np.concatenate([toks, prev[prev >= 0]])); remap = {int(t): j for j, t in enumerate(uniq)}; nu = len(uniq)
    tok_i = np.vectorize(lambda t: remap[int(t)])(toks).astype(np.int64)
    prev_i = np.array([remap[int(t)] if t >= 0 else 0 for t in prev], dtype=np.int64)
    # topic labels at each (seq,pos<=SEQ-2): cluster L15 content residual
    capL = []
    def hc(mo, i_, o_): capL.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh = m.transformer.h[CONTENT_L].register_forward_hook(hc)
    for i in range(0, nb, 8):
        forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    hh.remove()
    R15 = torch.cat(capL, 0)
    Utok, g = mean_subspace(R15, toks, RTOK)
    posarr = np.broadcast_to(np.arange(SEQ-1), (nb, SEQ-1)).reshape(-1)
    Upos, _ = mean_subspace(R15, posarr.astype(np.int64), RPOS)
    Ucp = torch.linalg.svd(torch.cat([Utok, Upos], 1), full_matrices=False)[0][:, :RTOK+RPOS].contiguous()
    content = (R15-g) - ((R15-g)@Ucp)@Ucp.T; cn = content/(content.norm(dim=1, keepdim=True)+1e-9)
    topic_i = kmeans(cn, K).cpu().numpy().astype(np.int64)
    rng = np.random.RandomState(0); topic_sh = topic_i.copy(); rng.shuffle(topic_sh)
    # capture each component output, build additive tables
    tags = [f"{k}{L}" for (L, k) in COMPS]; hooks = [submod(L, k).register_forward_hook(repl_hook_factory(f"{k}{L}")) for (L, k) in COMPS]
    cap = {}
    caph = []
    for (L, k) in COMPS:
        tag = f"{k}{L}"
        def mk(tag):
            def h(mo, i_, o_): cap.setdefault(tag, []).append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        caph.append(submod(L, k).register_forward_hook(mk(tag)))
    REPL['mode'] = 'off'
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in caph: h.remove()
    outs = {t: torch.cat(cap[t], 0) for t in tags}       # (nb*(SEQ-1), D) each
    # build per-component additive tables + precomputed per-position stand-in tensors for each level
    def build_standins(O):
        tt = table(O, tok_i, nu); r1 = O - tt[torch.tensor(tok_i, device=DEV)]
        tp = table(r1, topic_i, K); tp_sh = table(r1, topic_sh, K)
        r2 = r1 - tp[torch.tensor(topic_i, device=DEV)]; pv = table(r2, prev_i, nu)
        s_tok = tt[torch.tensor(tok_i, device=DEV)]
        s_tt = s_tok + tp[torch.tensor(topic_i, device=DEV)]
        s_ttp = s_tt + pv[torch.tensor(prev_i, device=DEV)]
        s_tsh = s_tok + tp_sh[torch.tensor(topic_sh, device=DEV)]
        gm = O.mean(0)
        return {'token': s_tok, 'token+topic': s_tt, 'token+topic+prev': s_ttp, 'token+shuftopic': s_tsh, 'gmean': gm}
    standins = {t: build_standins(outs[t]) for t in tags}
    # full CE
    REPL['mode'] = 'off'; ce_full = ce_under(blocks)
    def ce_with(tag, level):
        REPL['target'] = tag
        if level == 'mean':
            REPL['mode'] = 'mean'; REPL['gmean'] = standins[tag]['gmean']
        else:
            REPL['mode'] = 'set'
        tot = 0.0; n = 0; row = 0
        for i in range(0, nb, 8):
            bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
            bsz = idx.shape[0]
            if level != 'mean':
                REPL['val'] = standins[tag][level][row:row+bsz*(SEQ-1)].reshape(bsz, SEQ-1, D)
            lg = forward_logits(idx).float(); tot += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1), reduction='sum')); n += tgt.numel(); row += bsz*(SEQ-1)
        REPL['mode'] = 'off'; return tot / n
    out = {'ce_full': round(ce_full, 3), 'components': {}}
    for (L, k) in COMPS:
        tag = f"{k}{L}"; ce_mean = ce_with(tag, 'mean'); denom = max(ce_mean - ce_full, 1e-6)
        levels = {}
        for lv in ['token', 'token+topic', 'token+topic+prev', 'token+shuftopic']:
            ce = ce_with(tag, lv); levels[lv] = {'ce': round(ce, 3), 'frac': round(float((ce_mean - ce)/denom), 3)}
        levels['topic_genuine'] = round(levels['token+topic']['frac'] - levels['token+shuftopic']['frac'], 3)
        levels['topic_gain_over_token'] = round(levels['token+topic']['frac'] - levels['token']['frac'], 3)
        out['components'][tag] = {'ce_meanablate': round(ce_mean, 3), **levels}
        print(f"{tag:>6}: token {levels['token']['frac']:.2f} -> +topic {levels['token+topic']['frac']:.2f} (gain {levels['topic_gain_over_token']:+.2f}, genuine {levels['topic_genuine']:+.2f}) -> +prev {levels['token+topic+prev']['frac']:.2f}", flush=True)
    for h in hooks: h.remove()
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\n(0=mean-ablate,1=full; additive token→+topic→+prev stand-ins; genuine vs shuffled-topic)", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
