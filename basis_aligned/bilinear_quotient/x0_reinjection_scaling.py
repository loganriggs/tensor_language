"""The x0 EMBEDDING RE-INJECTION: each block computes x = lambda0*x + lambda1*x0, and lambda1 is HUGE/saturated
(mostly 8.0) vs lambda0 (~0.5-1.5) — so every block's input is DOMINATED by the re-injected original embedding
(~8/9 embedding + ~1/9 accumulated computation). This is why the token is ever-present and class is re-derived each
block (§962). Characterize its role gradedly: scale lambda1 by alpha and measure the chain-rule CE split. Compare
its content/grammar damage ratio to the VALUE RESIDUAL's (§985: 3.84, content-weighted). Hypothesis: x0
re-injection is the TOKEN/GRAMMAR substrate (keeps the token available for class re-derivation), so it should be
LESS content-weighted than the value residual.

REGISTERED PREDICTIONS:
  (0) SANITY: alpha=1 reproduces full CE; lambda1 is large so scaling it down is impactful (graded degradation).
  (a) x0 RE-INJECTION IS TOKEN/GRAMMAR-WEIGHTED (relative to the value residual): scaling lambda1 down hurts CE,
      and its content/grammar damage ratio is LOWER than the value residual's 3.84 (i.e. relatively MORE grammar
      damage) -> x0 re-injection keeps the token available for class re-derivation (§962), complementary to the
      value residual (content §985). Both are "keep the original token available" mechanisms with different
      downstream roles;
  (b) report class-CE and within-CE across alpha + the content/grammar damage ratio at alpha=0."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'x0_reinjection_scaling_results.json'
NEVAL = 160; SEQ = 256; ALPHAS = [1.0, 0.75, 0.5, 0.25, 0.0]
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
    l1_vals = [round(float(m.transformer.h[L].lambdas[1]),3) for L in range(18)]
    orig = [m.transformer.h[L].lambdas.data.clone() for L in range(18)]
    out = {'lambda1_per_layer': l1_vals, 'by_alpha': {}}
    for a in ALPHAS:
        for L in range(18):
            m.transformer.h[L].lambdas.data.copy_(orig[L]); m.transformer.h[L].lambdas.data[1] = orig[L][1]*a
        out['by_alpha'][str(a)] = split_ce(blocks, cidx, C)
        print(f"alpha {a}: {out['by_alpha'][str(a)]}", flush=True)
    for L in range(18): m.transformer.h[L].lambdas.data.copy_(orig[L])
    f = out['by_alpha']['1.0']; z = out['by_alpha']['0.0']
    out['class_ce_gain_at0'] = round(z['class_ce']-f['class_ce'], 4); out['within_ce_gain_at0'] = round(z['within_ce']-f['within_ce'], 4)
    out['content_vs_grammar_ratio_at0'] = round(out['within_ce_gain_at0']/max(abs(out['class_ce_gain_at0']),1e-6), 2)
    out['value_residual_ratio_ref_985'] = 3.84
    wc = [out['by_alpha'][str(a)]['within_ce'] for a in ALPHAS]; cc = [out['by_alpha'][str(a)]['class_ce'] for a in ALPHAS]
    out['pred_a_x0_more_grammar_weighted'] = bool(out['content_vs_grammar_ratio_at0'] < 3.84 and (z['full_ce']-f['full_ce']) > 0.1)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"lambda1: {l1_vals}", flush=True)
    print(f"within-CE across alpha(1->0): {wc} | class-CE: {cc}", flush=True)
    print(f"at alpha=0: grammar gain +{out['class_ce_gain_at0']} content gain +{out['within_ce_gain_at0']} ratio {out['content_vs_grammar_ratio_at0']} (value-residual ref 3.84)", flush=True)
    print(f"(a) x0 re-injection more grammar-weighted than value residual: {out['pred_a_x0_more_grammar_weighted']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
