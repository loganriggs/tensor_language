"""LOCALIZE where the MULTIPLICATIVE content is computed (multiplicative analog of §998, which localized content
POOLING to L3-5). §1000 gave the clean compositional-linearization content cost by band (front 1.01, middle 0.65,
all 1.59 nats). Here break it into 3-layer groups: compositionally linearize ONLY that group (each layer fit on the
already-linearized upstream) and measure the within-CE (content) cost. This maps which layers do the irreducible
multiplicative content computation.

REGISTERED PREDICTIONS:
  (0) CONSISTENCY: per-group within-costs are each < the §1000 whole-band costs; the sum of the six group costs is
      of the same order as (not wildly above) the all-band §1000 cost 1.59 (some super-additivity expected from the
      cooperative front, §994/§1000, so sum may exceed 1.59 modestly).
  (a) MULTIPLICATIVE CONTENT IS EARLY/EARLY-MIDDLE: the largest within-CE cost group is in L0-8 (the high-magnitude
      front writers + the content-gathering band L3-5), and the late groups (L12-17) are cheap -> the irreducible
      multiplicative content is computed early/early-middle, not at the readout;
  (b) report per-3-layer-group within-CE and class-CE cost."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_mult_by_group_results.json'
NCAL = 96; NEVAL = 160; SEQ = 256; RIDGE = 10.0
GROUPS = [(0,2),(3,5),(6,8),(9,11),(12,14),(15,17)]; ALLL = list(range(18))
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
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
        x = (i_[0] if isinstance(i_, tuple) else i_).float()
        o = o_[0] if isinstance(o_, tuple) else o_
        if L == CTX['capture']:
            CTX['buf'][0].append(x.reshape(-1, D).detach().cpu()); CTX['buf'][1].append(o.float().reshape(-1, D).detach().cpu())
            return None  # let TRUE output flow while capturing
        if L in CTX['installed']:
            x1 = torch.cat([x.reshape(-1, D), torch.ones(x.reshape(-1, D).shape[0], 1, device=DEV)], 1)
            return (x1 @ WLIN[L]).reshape(o.shape).to(o.dtype)
        return None
    return h


@torch.no_grad()
def fit_band_compositional(calib, band):
    CTX['installed'] = set()
    for L in band:  # ascending order matters: upstream fitted before downstream
        CTX['capture'] = L; CTX['buf'] = ([], [])
        for i in range(0, calib.shape[0], 8):
            forward_logits(calib[i:i+8].to(DEV)[:, :-1].contiguous())
        X = torch.cat(CTX['buf'][0], 0).to(DEV); Y = torch.cat(CTX['buf'][1], 0).to(DEV)
        n = min(X.shape[0], 12000)
        if X.shape[0] > n:
            sel = torch.randperm(X.shape[0], device=DEV)[:n]; X = X[sel]; Y = Y[sel]
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
    full = tot/n; classce = totc/n
    return {'full_ce': round(full, 4), 'class_ce': round(classce, 4), 'within_ce': round(full-classce, 4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NCAL + NEVAL); d = dec()
    calib = rows[:NCAL, :SEQ].contiguous(); blocks = rows[NCAL:NCAL+NEVAL, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    hooks = [m.transformer.h[L].mlp.register_forward_hook(hook_factory(L)) for L in ALLL]
    CTX['installed'] = set(); CTX['capture'] = None
    base = split_ce(blocks, cidx, C); print(f"baseline {base}", flush=True)
    out = {'baseline': base, 'conditions': {}}
    for a, b in GROUPS:
        band = list(range(a, b+1)); tag = f'L{a}_{b}'
        WLIN.clear(); fit_band_compositional(calib, band)
        CTX['installed'] = set(band); CTX['capture'] = None
        r = split_ce(blocks, cidx, C)
        r['within_cost'] = round(r['within_ce'] - base['within_ce'], 4)
        r['class_cost'] = round(r['class_ce'] - base['class_ce'], 4)
        out['conditions'][tag] = r
        CTX['installed'] = set()
        print(f"{tag:>7}: within-cost +{r['within_cost']} class-cost +{r['class_cost']} (full {r['full_ce']})", flush=True)
    for h in hooks: h.remove()
    wc = {t: out['conditions'][t]['within_cost'] for t in out['conditions']}
    top = max(wc, key=wc.get); early = sum(wc[f'L{a}_{b}'] for a,b in GROUPS if b <= 8); late = sum(wc[f'L{a}_{b}'] for a,b in GROUPS if a >= 12)
    out['top_group'] = top; out['early_L0_8_sum'] = round(early,4); out['late_L12_17_sum'] = round(late,4); out['group_sum'] = round(sum(wc.values()),4)
    out['pred_a_mult_early'] = bool(top in ('L0_2','L3_5','L6_8') and early > late)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top multiplicative-content group {top} | early(0-8) {early:.3f} late(12-17) {late:.3f} sum {out['group_sum']}", flush=True)
    print(f"pred_a multiplicative-content early {out['pred_a_mult_early']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
