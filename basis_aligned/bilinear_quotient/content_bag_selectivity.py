"""FRONTIER (content machine): content is a broad, long-range, order-invariant bag-of-words topic gist (§967/§995).
WHICH tokens are in the bag? Does the model pool CONTENT words (nouns/topic words) and ignore function words, or is
it a bag of ALL tokens? This refines the mechanism and tells the benchmark's content stand-in which tokens to average.

Restrict the FAR context (beyond a local window W kept always, so grammar stays intact) to only CONTENT-word key
positions vs only FUNCTION-word key positions, via a key-class mask on the attention pattern (monkeypatch every
layer's squared_attention). content words = {word, cap, number}; function words = {det, prep, conj, pron, punct}.

Conditions (local window W=4 tokens always kept; masking applies to keys at distance > W):
  all           : full causal (baseline)
  content_far   : far keys kept only if content-word
  function_far  : far keys kept only if function-word
  none_far      : far keys all dropped (only local window; upper bound on how much far context matters)

REGISTERED PREDICTIONS:
  (0) NULL: 'all' reproduces the true baseline CE (mask off).
  (a) BAG OF CONTENT WORDS: content_far preserves within-CE (content) MUCH better than function_far -- i.e. the
      within-CE cost of function_far is close to none_far (dropping far context), while content_far recovers most of
      the far-context content -> the bag is a bag of CONTENT words;
  (b) GRAMMAR LOCAL CONTROL: class-CE (grammar) is nearly unchanged across all conditions (local window preserved);
  (c) report within-CE and class-CE for each condition + the content-word share of far-context content recovery
      = (within_none - within_content)/(within_none - within_all)."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tt_model as TT
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_bag_selectivity_results.json'
NEVAL = 160; SEQ = 256; W = 4
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
CONTENT_CLS = {'word', 'cap', 'number'}
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
CTX = {'mode': 'all', 'keycontent': None}  # keycontent: (B,T) bool = key position is a content word


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


def masked_squared_attention(self, q, k, v, q2, k2):
    B, T, H, Dh = q.shape
    scores = torch.einsum('bqhd,bkhd->bhqk', q, k)
    scores2 = torch.einsum('bqhd,bkhd->bhqk', q2, k2)
    pattern = (scores / Dh) * (scores2 / Dh)
    i = torch.arange(T, device=pattern.device)
    causal = i[:, None] >= i[None, :]                 # (T,T)
    local = (i[:, None] - i[None, :]) < W             # (T,T) within local window (incl self)
    keep = causal.unsqueeze(0).expand(B, T, T).clone()  # (B,Tq,Tk)
    m_ = CTX['mode']
    if m_ != 'all':
        kc = CTX['keycontent']                        # (B,T) bool content-word at key pos
        if m_ == 'content_far':   farok = kc[:, None, :]
        elif m_ == 'function_far': farok = ~kc[:, None, :]
        else:                      farok = torch.zeros(B, 1, T, dtype=torch.bool, device=pattern.device)  # none_far
        keep = keep & (local[None] | farok)           # keep local always; far only if farok
    pattern = pattern.masked_fill(~keep.unsqueeze(1), 0.0)  # broadcast over heads
    z = torch.einsum('bhqk,bkhd->bhqd', pattern, v)
    return z


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def split_ce(blocks, cidx, C, iscontent):
    Cmat = F.one_hot(cidx, C).float(); tot = 0.0; totc = 0.0; n = 0
    for i in range(0, blocks.shape[0], 8):
        bb = blocks[i:i+8].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        CTX['keycontent'] = iscontent[idx]  # (B,T-1) content-word mask for the KEY positions of this batch
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
    tok2cls = np.full(V, 7, np.int64); tok2content = np.zeros(V, bool)
    for tid in np.unique(blocks.cpu().numpy().reshape(-1)):
        cls = classify(d(int(tid))); tok2cls[int(tid)] = CLASSES.index(cls); tok2content[int(tid)] = cls in CONTENT_CLS
    cidx = torch.tensor(tok2cls, device=DEV); iscontent = torch.tensor(tok2content, device=DEV)
    # true baseline (unpatched)
    base_orig = split_ce(blocks, cidx, C, iscontent)
    print(f"baseline (original) {base_orig}", flush=True)
    attns = [m.transformer.h[L].attn for L in range(18)]
    for a in attns: a.squared_attention = types.MethodType(masked_squared_attention, a)
    out = {'baseline_orig': base_orig, 'conditions': {}}
    for mode in ['all', 'content_far', 'function_far', 'none_far']:
        CTX['mode'] = mode
        r = split_ce(blocks, cidx, C, iscontent); out['conditions'][mode] = r
        print(f"{mode:>13}: {r}", flush=True)
    for a in attns: a.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, a)
    co = out['conditions']
    w_all = co['all']['within_ce']; w_none = co['none_far']['within_ce']
    w_content = co['content_far']['within_ce']; w_function = co['function_far']['within_ce']
    span = max(w_none - w_all, 1e-9)
    out['content_word_share'] = round((w_none - w_content) / span, 3)   # fraction of far content recovered by content words
    out['function_word_share'] = round((w_none - w_function) / span, 3)
    out['class_ce_spread'] = round(max(co[k]['class_ce'] for k in co) - min(co[k]['class_ce'] for k in co), 4)
    out['pred_0_null_ok'] = bool(abs(co['all']['full_ce'] - base_orig['full_ce']) < 0.01)
    out['pred_a_bag_of_content'] = bool(out['content_word_share'] > out['function_word_share'] + 0.2 and out['content_word_share'] > 0.5)
    out['pred_b_grammar_local_control'] = bool(out['class_ce_spread'] < 0.1)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"within: all {w_all} content_far {w_content} function_far {w_function} none_far {w_none}", flush=True)
    print(f"content-word share {out['content_word_share']} vs function-word share {out['function_word_share']} | class-CE spread {out['class_ce_spread']}", flush=True)
    print(f"pred_a bag-of-content {out['pred_a_bag_of_content']} | pred_b grammar-local-control {out['pred_b_grammar_local_control']} | null {out['pred_0_null_ok']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
