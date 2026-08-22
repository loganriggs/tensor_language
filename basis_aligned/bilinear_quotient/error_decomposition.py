"""ERROR INTERPRETABILITY via the two-machine account: when bilin18 is WRONG, is it a GRAMMAR failure (predicts the
wrong part-of-speech) or a CONTENT failure (right class, wrong specific word)? Decompose every next-token
prediction into: HIT (top-1 correct), CONTENT-ERROR (predicted class == true next-token class, but wrong token),
GRAMMAR-ERROR (predicted class != true class). Report frequency and CE-loss share of each. Extends the loss budget
(§831/§880: grammar easy 23% / content hard 77%) to the model's actual mistakes.

REGISTERED PREDICTIONS:
  (0) SANITY: HIT rate is well above chance; the three categories partition all positions.
  (a) CONTENT DOMINATES ERRORS: among errors, CONTENT-errors (right class, wrong word) are MORE frequent and carry
      MORE of the total CE loss than GRAMMAR-errors (wrong class) -> the model mostly gets the part-of-speech right
      and fails on the specific content word, consistent with grammar being the solved/easy machine and content
      the hard frontier;
  (b) report count, fraction, and mean-CE for HIT / CONTENT-ERROR / GRAMMAR-ERROR; also the loss share."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'error_decomposition_results.json'
NEVAL = 200; SEQ = 256
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
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); nb = blocks.shape[0]
    # token -> class table over the vocab that appears
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, 7, np.int64)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    hit = 0; cont_err = 0; gram_err = 0; n = 0
    ce_hit = 0.0; ce_cont = 0.0; ce_gram = 0.0
    for i in range(0, nb, 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); lp = F.log_softmax(lg, -1)
        tf = tgt.reshape(-1); lpf = lp.reshape(-1, lp.shape[-1])
        pred = lpf.argmax(1)
        ce = -lpf[torch.arange(tf.shape[0], device=DEV), tf]
        true_cls = cidx[tf]; pred_cls = cidx[pred]
        is_hit = pred == tf
        is_cont = (~is_hit) & (pred_cls == true_cls)
        is_gram = (~is_hit) & (pred_cls != true_cls)
        hit += int(is_hit.sum()); cont_err += int(is_cont.sum()); gram_err += int(is_gram.sum()); n += tf.shape[0]
        ce_hit += float(ce[is_hit].sum()); ce_cont += float(ce[is_cont].sum()); ce_gram += float(ce[is_gram].sum())
    tot_ce = ce_hit + ce_cont + ce_gram
    out = {'n': n,
           'hit': {'count': hit, 'frac': round(hit/n, 4), 'mean_ce': round(ce_hit/max(hit,1), 4), 'loss_share': round(ce_hit/tot_ce, 4)},
           'content_error': {'count': cont_err, 'frac': round(cont_err/n, 4), 'mean_ce': round(ce_cont/max(cont_err,1), 4), 'loss_share': round(ce_cont/tot_ce, 4)},
           'grammar_error': {'count': gram_err, 'frac': round(gram_err/n, 4), 'mean_ce': round(ce_gram/max(gram_err,1), 4), 'loss_share': round(ce_gram/tot_ce, 4)}}
    out['error_content_vs_grammar'] = {'content_frac_of_errors': round(cont_err/max(cont_err+gram_err,1), 3),
                                       'content_loss_share_of_errors': round(ce_cont/max(ce_cont+ce_gram,1e-9), 3)}
    out['pred_a_content_dominates_errors'] = bool(cont_err > gram_err and ce_cont > ce_gram)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"HIT {out['hit']['frac']} (loss share {out['hit']['loss_share']}) | CONTENT-err {out['content_error']['frac']} (share {out['content_error']['loss_share']}) | GRAMMAR-err {out['grammar_error']['frac']} (share {out['grammar_error']['loss_share']})", flush=True)
    print(f"of ERRORS: content {out['error_content_vs_grammar']['content_frac_of_errors']} by count, {out['error_content_vs_grammar']['content_loss_share_of_errors']} by loss", flush=True)
    print(f"(a) content dominates errors: {out['pred_a_content_dominates_errors']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
