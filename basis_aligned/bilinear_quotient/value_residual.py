"""What does the VALUE RESIDUAL do? Each block mixes the FIRST block's values into its attention:
v = (1-lamb)*v_current + lamb*v1, with a LEARNED per-layer scalar lamb ranging -4.2..+4.6 (heavily used; some
layers' values are DOMINATED by v1). Since v1 ~ the original token values, the value residual may be the mechanism
that makes content an order-invariant BAG of ORIGINAL word values (§932). Test: ablate the value residual (set all
lamb=0 -> v = v_current only) and measure the chain-rule CE split (class=grammar, within-class=content). If it
hurts CONTENT (within-CE) more than GRAMMAR (class-CE), the value residual is the bag-of-words content mechanism.

REGISTERED PREDICTIONS:
  (0) SANITY: ablating the value residual (lamb=0) changes CE (it is heavily used, |lamb| up to 4.6).
  (a) VALUE RESIDUAL = CONTENT MECHANISM: setting lamb=0 raises within-class (content) CE MORE than class
      (grammar) CE -> the value residual (pooling the first block's original token values v1) is the
      order-invariant bag-of-words CONTENT aggregation (§932); grammar is less affected;
  (b) report full vs lamb=0: class-CE, within-CE, full-CE, and the content-vs-grammar damage ratio + a per-layer
      lamb listing."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'value_residual_results.json'
NEVAL = 160; SEQ = 256
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','me','who','which'}


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


@torch.no_grad()
def split_ce(blocks, cidx, C):
    Cmat = F.one_hot(cidx, C).float(); tot=0.0; totc=0.0; n=0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(forward_logits(idx).float(), -1); tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        lp_tok = lpf[torch.arange(tf.shape[0], device=DEV), tf]
        pcls = (lpf.exp() @ Cmat).clamp_min(1e-12); lp_cls = pcls[torch.arange(tf.shape[0], device=DEV), cidx[tf]].log()
        tot += float(-lp_tok.sum()); totc += float(-lp_cls.sum()); n += tf.shape[0]
    full=tot/n; classce=totc/n; return {'full_ce': round(full,4), 'class_ce': round(classce,4), 'within_ce': round(full-classce,4)}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous()
    V = int(m.lm_head.weight.shape[0]); C = len(CLASSES)
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    lambs = [round(float(m.transformer.h[L].attn.lamb), 3) for L in range(18)]
    full = split_ce(blocks, cidx, C)
    # ablate value residual: set all lamb = 0
    orig = [m.transformer.h[L].attn.lamb.data.clone() for L in range(18)]
    for L in range(18): m.transformer.h[L].attn.lamb.data.zero_()
    abl = split_ce(blocks, cidx, C)
    for L in range(18): m.transformer.h[L].attn.lamb.data.copy_(orig[L])
    out = {'lamb_per_layer': lambs, 'full': full, 'value_residual_ablated': abl,
           'class_ce_gain': round(abl['class_ce'] - full['class_ce'], 4),
           'within_ce_gain': round(abl['within_ce'] - full['within_ce'], 4),
           'full_ce_gain': round(abl['full_ce'] - full['full_ce'], 4)}
    out['content_vs_grammar_damage_ratio'] = round(out['within_ce_gain'] / max(abs(out['class_ce_gain']), 1e-6), 2)
    out['pred_a_value_residual_is_content'] = bool(out['within_ce_gain'] > out['class_ce_gain'] and out['within_ce_gain'] > 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lamb per layer: {lambs}", flush=True)
    print(f"full: {full}", flush=True)
    print(f"lamb=0 (value-residual ablated): {abl}", flush=True)
    print(f"gains -> class(grammar) +{out['class_ce_gain']} | within(content) +{out['within_ce_gain']} | full +{out['full_ce_gain']} | content/grammar ratio {out['content_vs_grammar_damage_ratio']}", flush=True)
    print(f"(a) value residual is the content mechanism: {out['pred_a_value_residual_is_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
