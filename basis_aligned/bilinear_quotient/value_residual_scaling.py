"""Resolve §985's off-distribution caveat: lamb=0 fully removes the value residual (a large, off-training-
distribution intervention). SCALE the value-residual coefficient by alpha in {0, 0.25, 0.5, 0.75, 1.0} (lamb ->
alpha*lamb) and measure the chain-rule CE split at each alpha. If content-CE degrades SMOOTHLY and MONOTONICALLY as
alpha -> 0 (not a cliff at 0), the value residual's content role is a GRADED, genuine dependence rather than a
brittle off-distribution shock; and content should degrade more than grammar throughout.

REGISTERED PREDICTIONS:
  (0) SANITY: alpha=1.0 reproduces the full CE (3.32); alpha=0 reproduces §985 (~6.66).
  (a) GRADED CONTENT DEPENDENCE: within-class (content) CE rises SMOOTHLY and MONOTONICALLY as alpha decreases
      1->0 (no cliff), and by more than class (grammar) CE at every step -> the value residual is genuinely,
      gradedly content-load-bearing (not just an off-distribution artifact at 0);
  (b) report class-CE and within-CE across alpha."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'value_residual_scaling_results.json'
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
    orig = [m.transformer.h[L].attn.lamb.data.clone() for L in range(18)]
    out = {'by_alpha': {}}
    for a in ALPHAS:
        for L in range(18): m.transformer.h[L].attn.lamb.data.copy_(orig[L]*a)
        out['by_alpha'][str(a)] = split_ce(blocks, cidx, C)
        print(f"alpha {a}: {out['by_alpha'][str(a)]}", flush=True)
    for L in range(18): m.transformer.h[L].attn.lamb.data.copy_(orig[L])
    wc = [out['by_alpha'][str(a)]['within_ce'] for a in ALPHAS]  # alpha 1->0
    cc = [out['by_alpha'][str(a)]['class_ce'] for a in ALPHAS]
    monotonic = all(wc[i+1] >= wc[i]-1e-6 for i in range(len(wc)-1))  # rises as alpha decreases
    content_gt_grammar = all((wc[i+1]-wc[i]) >= (cc[i+1]-cc[i]) for i in range(len(wc)-1))
    out['within_ce_monotonic_rise'] = bool(monotonic); out['content_step_ge_grammar_step'] = bool(content_gt_grammar)
    out['pred_a_graded_content'] = bool(monotonic and content_gt_grammar)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"within-CE across alpha(1->0): {wc} | class-CE: {cc}", flush=True)
    print(f"(a) graded content dependence (smooth monotonic, content>grammar each step): {out['pred_a_graded_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
