"""ROBUSTNESS of the content/grammar instrument used throughout (§1000/§1005/§1010): the chain-rule class-CE/within-CE
split rests on a hand-coded 8-class taxonomy. Does the headline "linearizing the middle costs ~4x more CONTENT than
GRAMMAR" survive a DIFFERENT class granularity? Recompute the compositional middle-linearization cost split under
(a) the 8-class taxonomy, (b) a COARSE 2-class taxonomy (function words vs content words), and (c) a SHUFFLED-class
NULL (random token->class map, matched marginal sizes) for bilin18.

REGISTERED PREDICTIONS:
  (0) NULL: under a SHUFFLED class map the content/grammar cost ratio collapses toward the size-proportional baseline
      (no genuine content>grammar signal) -> the true split is not a partition-arithmetic artifact.
  (a) TAXONOMY-ROBUST: the content/grammar linearization-cost ratio is > 2.5 under BOTH the 8-class and the coarse
      2-class taxonomies (the "multiplication serves content" finding is not an artifact of the 8-class choice);
  (b) report within/class cost + ratio for 8-class, 2-class, and shuffled-null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mult_content_class_robust_results.json'
NCAL = 96; NEVAL = 160; SEQ = 256; RIDGE = 10.0; MIDDLE = list(range(6, 16))
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
FUNCTION = {'det', 'prep', 'conj', 'pron', 'punct'}   # coarse: function vs content
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
WLIN = {}; CTX = {'installed': set(), 'capture': None, 'buf': None}


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


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def hook_factory(L):
    def h(mo, i_, o_):
        x = (i_[0] if isinstance(i_, tuple) else i_).float(); o = o_[0] if isinstance(o_, tuple) else o_
        if L == CTX['capture']:
            CTX['buf'][0].append(x.reshape(-1, D).detach().cpu()); CTX['buf'][1].append(o.float().reshape(-1, D).detach().cpu())
            return None
        if L in CTX['installed']:
            x1 = torch.cat([x.reshape(-1, D), torch.ones(x.reshape(-1, D).shape[0], 1, device=DEV)], 1)
            return (x1 @ WLIN[L]).reshape(o.shape).to(o.dtype)
        return None
    return h


@torch.no_grad()
def fit_compositional(calib, band):
    CTX['installed'] = set()
    for L in band:
        CTX['capture'] = L; CTX['buf'] = ([], [])
        for i in range(0, calib.shape[0], 8): forward_logits(calib[i:i+8].to(DEV)[:, :-1].contiguous())
        X = torch.cat(CTX['buf'][0], 0).to(DEV); Y = torch.cat(CTX['buf'][1], 0).to(DEV)
        n = min(X.shape[0], 12000)
        if X.shape[0] > n: sel = torch.randperm(X.shape[0], device=DEV)[:n]; X = X[sel]; Y = Y[sel]
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
        WLIN[L] = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(D+1, device=DEV), X1.T @ Y); CTX['installed'].add(L); del X, Y
    CTX['capture'] = None


@torch.no_grad()
def split_ce(blocks, cidx, C):
    Cmat = F.one_hot(cidx, C).float(); tot = 0.0; totc = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tf.shape[0]
    return (tot-totc)/n, totc/n   # within, class


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL); d = dec()
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0])
    uniq = np.unique(blocks.cpu().numpy().reshape(-1))
    cls8 = np.full(V, 7, np.int64); cls2 = np.zeros(V, np.int64); rng = np.random.RandomState(0)
    for tid in uniq:
        c = classify(d(int(tid))); cls8[int(tid)] = CLASSES.index(c); cls2[int(tid)] = 0 if c in FUNCTION else 1
    clsSh = cls8.copy(); clsSh[uniq] = rng.permutation(cls8[uniq])  # shuffled 8-class labels among seen tokens
    maps = {'class8': (torch.tensor(cls8, device=DEV), 8), 'class2': (torch.tensor(cls2, device=DEV), 2),
            'shuffled8': (torch.tensor(clsSh, device=DEV), 8)}
    hooks = [m.transformer.h[L].mlp.register_forward_hook(hook_factory(L)) for L in range(18)]
    WLIN.clear(); CTX['installed'] = set(); CTX['capture'] = None
    # baselines per map
    base = {k: split_ce(blocks, cidx, C) for k, (cidx, C) in maps.items()}
    # linearize middle once (map-independent), then measure split under each map
    fit_compositional(calib, MIDDLE); CTX['installed'] = set(MIDDLE); CTX['capture'] = None
    out = {'conditions': {}}
    for k, (cidx, C) in maps.items():
        w, c = split_ce(blocks, cidx, C); bw, bc = base[k]
        wc = round(w-bw, 4); cc = round(c-bc, 4)
        out['conditions'][k] = {'within_cost': wc, 'class_cost': cc, 'ratio': round(wc/max(abs(cc), 1e-6), 2)}
        print(f"{k:>10}: within +{wc} class +{cc} ratio {out['conditions'][k]['ratio']}", flush=True)
    for h in hooks: h.remove()
    r8 = out['conditions']['class8']['ratio']; r2 = out['conditions']['class2']['ratio']; rs = out['conditions']['shuffled8']['ratio']
    out['pred_0_null_collapses'] = bool(rs < r8 - 1.0)
    out['pred_a_taxonomy_robust'] = bool(r8 > 2.5 and r2 > 2.5)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"ratios: 8-class {r8} | 2-class {r2} | shuffled-null {rs}", flush=True)
    print(f"pred_0 null-collapses {out['pred_0_null_collapses']} | pred_a taxonomy-robust {out['pred_a_taxonomy_robust']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
