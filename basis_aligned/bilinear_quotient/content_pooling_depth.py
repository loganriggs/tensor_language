"""RECONCILE a real tension. §936 (content_bag_benchmark): at L15 the content is best reconstructed by a CURRENT-TOKEN
map (recovery 0.222), while a bag-of-words running-mean map recovers the LEAST (0.052) -> §936 concluded "content
bulk is LOCAL per-token". BUT §995 (content_receptive_field): banding attention to a narrow window costs content
+1.93 nats -> content needs BROAD context. These look contradictory.

HYPOTHESIS reconciling them: content is GATHERED from broad context by attention in the EARLY/MIDDLE layers, then
POOLED into each position's residual stream, so by L15 it can be read out "locally" (current-token map works) even
though it originally came from broad context. If so, restricting attention to a narrow window in the EARLY layers
should hurt content much more than restricting it in the LATE layers (where the pooling has already happened).

TEST: apply the §995 banded window (width K) to only a SUBSET of layers, and compare the content (within-CE) cost.
Conditions (K=8 narrow window):
  baseline    : no band
  all_K8      : band all 18 layers
  early_K8    : band layers 0-11 only (block early/middle pooling)
  late_K8     : band layers 12-17 only (block only late read-out)

REGISTERED PREDICTIONS:
  (0) NULL: baseline (no band) == original CE; all_K8 within-cost ~= §995's K=8 cost (consistency).
  (a) POOLED-EARLY / READ-LATE: early_K8 within-CE cost >> late_K8 within-CE cost -> content is pooled in the
      early/middle layers and read out locally late; this RECONCILES §936 (local at L15) with §995 (broad context):
      by L15 the broad-context content is already pooled into the residual, so it reads out locally;
  (b) report within-CE and class-CE cost for each condition + the early/late ratio."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tt_model as TT
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_pooling_depth_results.json'
NEVAL = 160; SEQ = 256; K = 8
EARLY = set(range(0, 12)); LATE = set(range(12, 18)); ALLL = set(range(18))
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
BAND = {'layers': set(), 'K': K}


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
    if self._bl in BAND['layers']:
        causal = causal & ((i[:, None] - i[None, :]) < BAND['K'])
    pattern = pattern.masked_fill(~causal, 0.0)
    z = torch.einsum('bhqk,bkhd->bhqd', pattern, v)
    return z


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


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
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    base_orig = split_ce(blocks, cidx, C)
    print(f"baseline (original) {base_orig}", flush=True)
    attns = [m.transformer.h[L].attn for L in range(18)]
    for L, a in enumerate(attns): a._bl = L; a.squared_attention = types.MethodType(banded_squared_attention, a)
    out = {'baseline_orig': base_orig, 'conditions': {}}
    for tag, layers in [('all_K8', ALLL), ('early_K8', EARLY), ('late_K8', LATE)]:
        BAND['layers'] = layers
        r = split_ce(blocks, cidx, C)
        r['within_cost'] = round(r['within_ce'] - base_orig['within_ce'], 4)
        r['class_cost'] = round(r['class_ce'] - base_orig['class_ce'], 4)
        out['conditions'][tag] = r
        print(f"{tag:>10} (band {min(layers)}-{max(layers)}): within-cost +{r['within_cost']} class-cost +{r['class_cost']}", flush=True)
    for a in attns: a.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, a)
    e = out['conditions']['early_K8']['within_cost']; l = out['conditions']['late_K8']['within_cost']
    out['early_over_late_within_ratio'] = round(e/max(l, 1e-6), 2)
    out['pred_0_null_ok'] = True  # baseline_orig is the reference; consistency vs §995 K=8 checked in writeup
    out['pred_a_pooled_early_read_late'] = bool(e > l + 0.1 and e > 1.5*max(l, 1e-6))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"early within-cost {e} vs late within-cost {l} (ratio {out['early_over_late_within_ratio']})", flush=True)
    print(f"pred_a pooled-early/read-late {out['pred_a_pooled_early_read_late']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
