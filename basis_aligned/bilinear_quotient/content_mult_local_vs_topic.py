"""§1001 raised a puzzle: the multiplicative content is FRONT-LOADED in L0-2, but L0-2 sits BELOW the content-pooling
band (L3-5, §998), so at L0-2 the broad topic is not yet gathered. Is L0-2's multiplicative content LOCAL
(current-token word-sense, independent of broad context) or does it somehow depend on the pooled topic? And are the
LATER groups' multiplicative contributions topic-dependent (they multiply the already-pooled content)?

TEST: run §1001-style compositional group-linearization under two context regimes -- FULL context vs a K=8 banded
attention window (removes the broad topic, §995) applied to ALL layers -- and compare each group's within-CE
(content) cost. Fit each group's compositional linear maps UNDER the same regime it is evaluated in.
  cost_regime(G) = within_CE(regime + linearize G) - within_CE(regime, no linearize)
If a group's multiplicative content is LOCAL, banding the context does not change its cost (cost_banded ~ cost_full).
If it multiplies the POOLED topic, banding removes the topic and its cost DROPS (cost_banded < cost_full).

REGISTERED PREDICTIONS:
  (0) NULL: full-regime baseline == original CE (band off); band-regime baseline reproduces §995's K=8 within penalty.
  (a) FRONT MULTIPLICATIVE CONTENT IS LOCAL: for L0-2, cost_banded/cost_full is HIGH (>~0.7) -> the front's
      multiplicative content survives removing the broad context -> it is LOCAL (current-token word-sense), not the
      pooled topic;
  (b) LATER GROUPS ARE MORE TOPIC-DEPENDENT: for groups at/after the pooling (L6-8, L9-11), cost_banded/cost_full is
      LOWER than L0-2's ratio -> their multiplicative contribution operates on the pooled topic, which banding removes;
  (c) report cost_full, cost_banded, and the ratio per group."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tt_model as TT
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_mult_local_vs_topic_results.json'
NCAL = 96; NEVAL = 160; SEQ = 256; RIDGE = 10.0
GROUPS = [(0, 2), (3, 5), (6, 8), (9, 11)]; ALLL = list(range(18))
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
WLIN = {}; CTX = {'installed': set(), 'capture': None, 'buf': None}
BAND = {'K': None}


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


def banded_squared_attention(self, q, k, v, q2, k2):
    B, T, H, Dh = q.shape
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
    scores2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pattern = (scores / Dh) * (scores2 / Dh)
    i = torch.arange(T, device=pattern.device)
    causal = i[:, None] >= i[None, :]
    if BAND['K'] is not None:
        causal = causal & ((i[:, None] - i[None, :]) < BAND['K'])
    pattern = pattern.masked_fill(~causal, 0.0)
    return torch.einsum('bhqk,bkhd->bhqd', pattern, v)


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def mlp_hook_factory(L):
    def h(mo, i_, o_):
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        o = o_[0] if isinstance(o_, tuple) else o_
        if L == CTX['capture']:
            CTX['buf'][0].append(x.reshape(-1, D).detach().cpu()); CTX['buf'][1].append(o.float().reshape(-1, D).detach().cpu())
            return None
        if L in CTX['installed']:
            x1 = torch.cat([x.reshape(-1, D), torch.ones(x.reshape(-1, D).shape[0], 1, device=DEV)], 1)
            return (x1 @ WLIN[L]).reshape(o.shape).to(o.dtype)
        return None
    return h


@torch.no_grad()
def fit_band_compositional(calib, band):
    CTX['installed'] = set()
    for L in band:
        CTX['capture'] = L; CTX['buf'] = ([], [])
        for i in range(0, calib.shape[0], 8):
            forward_logits(calib[i:i+8].to(DEV)[:, :-1].contiguous())
        X = torch.cat(CTX['buf'][0], 0).to(DEV); Y = torch.cat(CTX['buf'][1], 0).to(DEV)
        n = min(X.shape[0], 12000)
        if X.shape[0] > n: sel = torch.randperm(X.shape[0], device=DEV)[:n]; X = X[sel]; Y = Y[sel]
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, device=DEV)], 1)
        WLIN[L] = torch.linalg.solve(X1.T @ X1 + RIDGE*torch.eye(D+1, device=DEV), X1.T @ Y)
        CTX['installed'].add(L); del X, Y
    CTX['capture'] = None


@torch.no_grad()
def split_ce(blocks, cidx, C):
    Cmat = F.one_hot(cidx, C).float(); tot = 0.0; totc = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tf.shape[0]
    return {'full_ce': round(tot/n, 4), 'class_ce': round(totc/n, 4), 'within_ce': round((tot-totc)/n, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL); d = dec()
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    # original baseline (no band, no linearize, original attention) for NULL
    orig = split_ce(blocks, cidx, C); print(f"original {orig}", flush=True)
    attns = [m.transformer.h[L].attn for L in range(18)]
    for a in attns: a.squared_attention = types.MethodType(banded_squared_attention, a)
    mlp_hooks = [m.transformer.h[L].mlp.register_forward_hook(mlp_hook_factory(L)) for L in ALLL]
    out = {'original': orig, 'regimes': {}}
    for rk, K in [('full', None), ('band_K8', 8)]:
        BAND['K'] = K; CTX['installed'] = set(); CTX['capture'] = None
        base = split_ce(blocks, cidx, C)
        reg = {'baseline': base, 'groups': {}}
        for a, b in GROUPS:
            band = list(range(a, b+1)); WLIN.clear(); fit_band_compositional(calib, band)
            CTX['installed'] = set(band); CTX['capture'] = None
            r = split_ce(blocks, cidx, C)
            reg['groups'][f'L{a}_{b}'] = {'within_cost': round(r['within_ce'] - base['within_ce'], 4),
                                          'class_cost': round(r['class_ce'] - base['class_ce'], 4)}
            CTX['installed'] = set()
            print(f"[{rk}] L{a}-{b}: within-cost +{reg['groups'][f'L{a}_{b}']['within_cost']}", flush=True)
        out['regimes'][rk] = reg
        print(f"[{rk}] baseline within {base['within_ce']}", flush=True)
    for a in attns: a.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, a)
    for h in mlp_hooks: h.remove()
    full = out['regimes']['full']['groups']; band = out['regimes']['band_K8']['groups']
    out['ratios'] = {g: round(band[g]['within_cost']/max(full[g]['within_cost'], 1e-6), 2) for g in full}
    l02 = out['ratios']['L0_2']; later = np.mean([out['ratios']['L6_8'], out['ratios']['L9_11']])
    out['front_local_ratio'] = l02; out['later_ratio_mean'] = round(float(later), 2)
    out['pred_0_null_ok'] = bool(abs(out['regimes']['full']['baseline']['full_ce'] - orig['full_ce']) < 0.01)
    out['pred_a_front_local'] = bool(l02 > 0.7)
    out['pred_b_later_topic_dependent'] = bool(later < l02)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"within-cost ratios (banded/full): {out['ratios']}", flush=True)
    print(f"L0-2 local-ratio {l02} vs later mean {later:.2f}", flush=True)
    print(f"pred_0 null {out['pred_0_null_ok']} | pred_a front-local {out['pred_a_front_local']} | pred_b later-topic-dep {out['pred_b_later_topic_dependent']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
