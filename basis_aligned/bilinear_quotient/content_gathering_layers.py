"""LOCALIZE the content-gathering layers (refines §997). §997 showed content is gathered in layers 0-11 (banding their
attention window costs +0.98 nats of content) and read out locally in 12-17 (+0.13). WHERE within 0-11 does the
gathering happen? Band the §995 attention window (K=8) on each 3-layer group in turn and measure the content
(within-CE) cost. This pinpoints the gathering band (hint: prior work noted content prediction "begins ~attn5").

REGISTERED PREDICTIONS:
  (0) NULL/consistency: baseline == original; the sum of per-group within-costs ~ the all-layers §997 cost (~1.1),
      and the 0-11 groups together dominate the 12-17 groups (consistency with §997).
  (a) GATHERING IS CONCENTRATED EARLY-MIDDLE: the largest content (within-CE) cost is in an early/early-middle
      3-layer group (0-2, 3-5, or 6-8), and the late groups (12-14, 15-17) are cheap -> content gathering is
      localized to the early/middle layers, not spread evenly;
  (b) report within-CE and class-CE cost per 3-layer group."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tt_model as TT
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_gathering_layers_results.json'
NEVAL = 160; SEQ = 256; K = 8
GROUPS = [(0,2),(3,5),(6,8),(9,11),(12,14),(15,17)]
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
    for a, b in GROUPS:
        BAND['layers'] = set(range(a, b+1)); tag = f'L{a}_{b}'
        r = split_ce(blocks, cidx, C)
        r['within_cost'] = round(r['within_ce'] - base_orig['within_ce'], 4)
        r['class_cost'] = round(r['class_ce'] - base_orig['class_ce'], 4)
        out['conditions'][tag] = r
        print(f"{tag:>7} (band {a}-{b}): within-cost +{r['within_cost']} class-cost +{r['class_cost']}", flush=True)
    for a in attns: a.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, a)
    wc = {t: out['conditions'][t]['within_cost'] for t in out['conditions']}
    top = max(wc, key=wc.get)
    early_sum = sum(wc[f'L{a}_{b}'] for a,b in GROUPS if b <= 11); late_sum = sum(wc[f'L{a}_{b}'] for a,b in GROUPS if a >= 12)
    out['top_group'] = top; out['early_sum'] = round(early_sum,4); out['late_sum'] = round(late_sum,4)
    out['pred_a_gathering_early_middle'] = bool(top in ('L0_2','L3_5','L6_8') and late_sum < early_sum)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"top content-gathering group {top} | early-sum {early_sum:.3f} late-sum {late_sum:.3f}", flush=True)
    print(f"pred_a gathering early-middle {out['pred_a_gathering_early_middle']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
