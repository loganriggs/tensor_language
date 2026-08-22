"""IS THE within-class loss FLOOR true irreducible entropy, or bilin18's LIMITATION? (settles the
§830 overclaim). §830 called bilin18's ~2.4-nat within-class (word-choice) CE floor "irreducible
entropy". That was an overclaim: it is bilin18's residual, not a measured entropy. Test: feed the
SAME FineWeb tokens (GPT-2 BPE, which bilin18 uses) to GPT-2 small/medium/large and measure their
within-class CE with the SAME class assignment. If within-class CE DROPS with model scale, the floor
is REDUCIBLE (capability-limited) — there is more word-choice information available and more to
understand; if it is FLAT across the 6x scale range, it is plausibly near-irreducible.

Caveat registered: the GPT-2 models were trained on WebText, not FineWeb, so they are mildly OOD here;
a scale-DROP is therefore a strong (conservative) signal for reducibility, while a flat result is
weaker (could be OOD). bilin18 (trained on FineWeb) is the in-distribution reference point.

REGISTERED PREDICTIONS:
  (0) SANITY: full CE and within-class CE computed on identical token blocks for all models;
  (a) REDUCIBLE: within-class CE drops monotonically gpt2 -> medium -> large (>=0.3 nats total) ->
      the word-choice floor is capability-limited, NOT irreducible entropy;
  (b) NEAR-IRREDUCIBLE: within-class CE ~flat across scale (<0.15 nats) -> consistent with the floor
      being close to irreducible on this data;
  report full CE and within-class CE for bilin18 + gpt2/medium/large on the same tokens."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m as bilin, DEV
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'entropy_ceiling_results.json'
D = 1152; NEVAL = 200
HF = ['gpt2', 'gpt2-medium', 'gpt2-large']
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}


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


def bilin_logits(idx):
    x = F.rms_norm(bilin.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in bilin.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(bilin.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def split_ce(logits_fn, blocks, cidx, Cmat, V):
    tc = tw = 0.0; n = 0
    for i in range(0, blocks.shape[0], 4):
        b = blocks[i:i+4].to(DEV); idx = b[:, :-1].contiguous(); tgt = b[:, 1:].contiguous()
        lg = logits_fn(idx).float(); logp = F.log_softmax(lg, -1); pcl = logp.exp() @ Cmat
        tgtf = tgt.reshape(-1); logpf = logp.reshape(-1, V); pcf = pcl.reshape(-1, len(CLASSES)); tgt_cls = cidx[tgtf]
        lp_tok = logpf[torch.arange(tgtf.shape[0], device=DEV), tgtf]
        lp_cls = (pcf[torch.arange(tgtf.shape[0], device=DEV), tgt_cls] + 1e-12).log()
        tc += float((-lp_cls).sum()); tw += float((-(lp_tok - lp_cls)).sum()); n += tgtf.shape[0]
    return tc/n, tw/n


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)                       # GPT-2 BPE token ids, 257 wide
    blocks = rows[:, :257]
    d = dec(); V = 50257
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(blocks.reshape(-1).cpu().numpy()):
        if tid < V: tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    out = {}
    # bilin18 (vocab may be 50304-padded; clamp Cmat/cidx to model vocab)
    Vb = int(bilin.lm_head.weight.shape[0])
    cidx_b = cidx if Vb == V else torch.tensor(np.concatenate([tok2cls, np.full(Vb-V, CLASSES.index('word'))]) if Vb > V else tok2cls[:Vb], device=DEV)
    Cmat_b = F.one_hot(cidx_b, len(CLASSES)).float()
    c, w = split_ce(bilin_logits, blocks, cidx_b, Cmat_b, Vb)
    out['bilin18'] = {'ce_class': round(c, 4), 'ce_within': round(w, 4), 'ce_total': round(c+w, 4)}
    print(f"bilin18: class {c:.3f} within {w:.3f} total {c+w:.3f}", flush=True)
    for mid in HF:
        mdl = AutoModelForCausalLM.from_pretrained(mid).to(DEV).eval()
        fn = lambda idx: mdl(idx).logits
        c, w = split_ce(fn, blocks, cidx, Cmat, V)
        out[mid] = {'ce_class': round(c, 4), 'ce_within': round(w, 4), 'ce_total': round(c+w, 4)}
        print(f"{mid}: class {c:.3f} within {w:.3f} total {c+w:.3f}", flush=True)
        del mdl; torch.cuda.empty_cache()
    wg = [out[m]['ce_within'] for m in HF]
    drop = round(wg[0] - wg[-1], 4)
    out['within_drop_gpt2_to_large'] = drop
    out['pred_a_reducible'] = bool(drop >= 0.3 and wg[0] > wg[1] > wg[2])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwithin-class CE across GPT-2 scale: {wg} (drop small->large {drop})", flush=True)
    print(f"(a) word-choice floor is REDUCIBLE with scale (not irreducible entropy): {out['pred_a_reducible']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
