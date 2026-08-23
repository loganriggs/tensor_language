"""FRONTIER (content machine): §994 reframed the content as living in the RESIDUAL STREAM, pooled by ATTENTION over
the context (bag-of-words, §931), not in the middle MLPs. So: HOW FAR BACK does the content the model actually USES
come from? Map the content machine's effective CONTEXT WINDOW by restricting every attention layer to a causal BAND
of width K (position i may attend only to keys in (i-K, i]), sweep K, and watch the chain-rule CE split recover.
This tells us what a content stand-in must integrate, and separates the grammar window (expected local) from the
content window (expected long-range).

Implementation: monkeypatch CausalBilinearSelfAttention.squared_attention on every layer with a verbatim copy that
adds a band mask (keys with i-j >= K are zeroed, on top of the causal mask). K=None restores the full causal model.

REGISTERED PREDICTIONS:
  (0) NULL: K=full (>=SEQ) reproduces baseline CE exactly (band mask off).
  (a) GRAMMAR IS LOCAL: class-CE (grammar) saturates at SMALL K (recovers ~fully by K<=4-8) -> grammar needs only
      the last few tokens (consistent with the front/local grammar machine);
  (b) CONTENT IS LONG-RANGE: within-CE (content) keeps improving well beyond K=8, needing LARGE K (tens-hundreds of
      tokens) to recover -> content is pooled over a long context (bag-of-words), NOT local;
  (c) report full/class/within CE at each K and the K at which each reaches 90% of its full-window recovery."""
import json, time, sys, types, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
sys.path.insert(0, '/workspace/tensor_language/jacclust')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tt_model as TT
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_receptive_field_results.json'
NEVAL = 160; SEQ = 256
KS = [1, 2, 4, 8, 16, 32, 64, 128, None]  # None = full causal (baseline/null)
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}
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
        causal = causal & ((i[:, None] - i[None, :]) < BAND['K'])  # keep keys within (i-K, i]
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
    base_orig = split_ce(blocks, cidx, C)  # unpatched original model (real null reference)
    print(f"baseline (original method) {base_orig}", flush=True)
    attns = [m.transformer.h[L].attn for L in range(18)]
    for a in attns: a.squared_attention = types.MethodType(banded_squared_attention, a)
    out = {'baseline_orig': base_orig, 'by_K': {}}
    for K in KS:
        BAND['K'] = K
        r = split_ce(blocks, cidx, C); out['by_K'][str(K)] = r
        print(f"K={str(K):>4}: {r}", flush=True)
    for a in attns: a.squared_attention = types.MethodType(TT.CausalBilinearSelfAttention.squared_attention, a)
    full = out['by_K']['None']; k1 = out['by_K']['1']
    # recovery fraction toward full, per K, for class and within (0 at K=1, 1 at K=full)
    def rec(metric):
        base1 = k1[metric]; span = base1 - full[metric]
        return {str(K): round((base1 - out['by_K'][str(K)][metric]) / max(span, 1e-9), 3) for K in KS}
    out['class_recovery'] = rec('class_ce'); out['within_recovery'] = rec('within_ce')

    def k90(metric):
        span = k1[metric] - full[metric]
        for K in KS:
            if K is None: return None
            if (k1[metric] - out['by_K'][str(K)][metric]) >= 0.9*span: return K
        return None
    out['class_K90'] = k90('class_ce'); out['within_K90'] = k90('within_ce')
    out['pred_0_null_ok'] = bool(abs(out['by_K']['None']['full_ce'] - base_orig['full_ce']) < 0.01)  # patched full band == original
    ck = out['class_K90']; wk = out['within_K90']
    out['pred_a_grammar_local'] = bool(ck is not None and ck <= 8)
    out['pred_b_content_longrange'] = bool(wk is not None and wk >= 16 and (ck is None or wk > ck))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"class recovery {out['class_recovery']}", flush=True)
    print(f"within recovery {out['within_recovery']}", flush=True)
    print(f"class K90 {ck} | within K90 {wk}", flush=True)
    print(f"pred_a grammar-local {out['pred_a_grammar_local']} | pred_b content-longrange {out['pred_b_content_longrange']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
