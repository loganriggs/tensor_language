"""WHAT is the 'irreducible' loss, concretely? (directly answers the user's question about irreducible
entropy — where it happens, what the evidence is, whether there's nothing to understand there). §829/§840:
the content/word-choice loss is large (~2.4 nats) and only partly reducible with scale. Here we localize it
token-by-token by comparing bilin18 to a MUCH larger reference model (gpt2-large, ~770M, same GPT-2 BPE
vocab so per-token losses align) on the SAME FineWeb tokens:
  - SHARED FLOOR (large-model loss): an UPPER BOUND on the irreducible entropy — hard even for a big model.
  - REDUCIBLE-BY-CAPACITY (bilin18 loss - large loss where positive): provably closable with scale.
Then characterize WHICH tokens sit in the shared floor (class, is-this-token's-first-occurrence-in-context,
position) with concrete examples, to say what 'irreducible' actually consists of.

CAVEAT: gpt2-large was trained on WebText, bilin18 on FineWeb, so large is slightly out-of-distribution
(inflates its absolute loss a little); the CHARACTER of shared-high-loss tokens is robust to that level
shift, and is the main result.

REGISTERED PREDICTIONS:
  (0) SANITY: large mean CE < bilin18 mean CE (bigger is better); per-token losses positively correlated;
  (a) THE FLOOR IS OPEN CONTENT CHOICE: tokens high-loss for BOTH models are dominated by CONTENT classes
      (word/cap/number) and especially FIRST-occurrences (a token type not yet seen in the context —
      new entities/names/word choices), NOT function words -> 'irreducible' = the specific word within a set
      topic, which is genuinely open; whereas the bilin18-only gap (bilin18 high, large low) is LESS
      first-mention / more predictable-in-principle (bilin18's capacity limit, not irreducible);
  (b) quantify: mean bilin18 CE, mean large CE, and the fraction of bilin18's loss that is shared-floor
      (min of the two) vs bilin18-specific reducible. Report class + first-mention breakdown for the top
      loss quartile of each model, with examples."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'irreducible_tokens_results.json'
NEVAL = 200; SEQ = 256
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'who', 'which'}
FUNCTION = {'det', 'prep', 'conj', 'pron', 'punct'}


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
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def per_token_loss_bilin(blocks):
    losses = []
    for i in range(0, blocks.shape[0], 4):
        bb = blocks[i:i+4].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lp = F.log_softmax(bilin_logits(idx).float(), -1)
        l = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        losses.append(l.cpu())
    return torch.cat(losses, 0)   # (nb, SEQ-1)


@torch.no_grad()
def per_token_loss_hf(mdl, blocks):
    losses = []
    for i in range(0, blocks.shape[0], 4):
        bb = blocks[i:i+4].to(DEV)
        lg = mdl(bb).logits.float(); lp = F.log_softmax(lg[:, :-1], -1); tgt = bb[:, 1:]
        l = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        losses.append(l.cpu())
    return torch.cat(losses, 0)


def breakdown(mask, S_next, first_next, cls_next, d, name):
    """class dist + first-mention frac + examples for the token positions in mask (flat)."""
    idx = np.where(mask)[0]
    cd = {c: 0 for c in CLASSES}
    for c in cls_next[idx]: cd[CLASSES[c]] += 1
    tot = max(len(idx), 1)
    cd = {c: round(v/tot, 3) for c, v in cd.items()}
    func_frac = round(sum(cd[c] for c in FUNCTION), 3)
    first_frac = round(float(first_next[idx].mean()), 3) if len(idx) else 0.0
    exs = [repr(d(int(S_next[j]))) for j in idx[:12]]
    return {'n': int(len(idx)), 'class_dist': cd, 'function_frac': func_frac, 'content_frac': round(1-func_frac, 3),
            'first_mention_frac': first_frac, 'examples': exs}


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous()
    Lb = per_token_loss_bilin(blocks)                       # (nb, SEQ-1)
    print("loading gpt2-large...", flush=True)
    large = AutoModelForCausalLM.from_pretrained('gpt2-large').to(DEV).eval()
    Ll = per_token_loss_hf(large, blocks)
    del large; torch.cuda.empty_cache()
    S = blocks.cpu().numpy()                                # (nb, SEQ)
    nb = S.shape[0]
    S_next = S[:, 1:].reshape(-1); Lb = Lb.numpy().reshape(-1); Ll = Ll.numpy().reshape(-1)
    # first-occurrence: is this next-token's id absent from the context before it (within the sequence)?
    first = np.zeros_like(S_next, dtype=bool)
    for r in range(nb):
        seen = set()
        for p in range(SEQ-1):
            tid = int(S[r, p+1]); first[r*(SEQ-1)+p] = tid not in seen
            seen.add(int(S[r, p]))
    cls_next = np.array([CLASSES.index(classify(d(int(t)))) for t in S_next])
    # stats
    corr = float(np.corrcoef(Lb, Ll)[0, 1])
    gap = Lb - Ll                                           # bilin18 minus large
    shared_floor = np.minimum(Lb, Ll)
    out = {'n_tokens': int(len(S_next)),
           'bilin18_mean_ce': round(float(Lb.mean()), 3), 'large_mean_ce': round(float(Ll.mean()), 3),
           'mean_gap_bilin_minus_large': round(float(gap.mean()), 3),
           'mean_shared_floor': round(float(shared_floor.mean()), 3),
           'shared_floor_frac_of_bilin': round(float(shared_floor.mean()/Lb.mean()), 3),
           'loss_correlation': round(corr, 3),
           'first_mention_mean_ce_bilin': round(float(Lb[first].mean()), 3),
           'seen_mean_ce_bilin': round(float(Lb[~first].mean()), 3),
           'first_mention_mean_ce_large': round(float(Ll[first].mean()), 3),
           'seen_mean_ce_large': round(float(Ll[~first].mean()), 3)}
    # top-quartile masks
    qb = np.quantile(Ll, 0.75)                              # high for the LARGE model = shared floor
    shared_high = Ll >= qb
    bilin_only = (Lb >= np.quantile(Lb, 0.75)) & (Ll < np.quantile(Ll, 0.5))   # bilin high, large low
    out['shared_floor_tokens'] = breakdown(shared_high, S_next, first, cls_next, d, 'shared_floor')
    out['bilin_only_gap_tokens'] = breakdown(bilin_only, S_next, first, cls_next, d, 'bilin_only')
    out['pred_a_floor_is_open_content'] = bool(
        out['shared_floor_tokens']['content_frac'] > out['bilin_only_gap_tokens']['content_frac'] and
        out['shared_floor_tokens']['first_mention_frac'] > out['bilin_only_gap_tokens']['first_mention_frac'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"bilin18 mean CE {out['bilin18_mean_ce']} | gpt2-large mean CE {out['large_mean_ce']} | gap {out['mean_gap_bilin_minus_large']} | corr {corr:.3f}", flush=True)
    print(f"shared floor (min) {out['mean_shared_floor']} = {out['shared_floor_frac_of_bilin']} of bilin18's loss", flush=True)
    print(f"first-mention CE: bilin {out['first_mention_mean_ce_bilin']} vs seen {out['seen_mean_ce_bilin']} | large {out['first_mention_mean_ce_large']} vs seen {out['seen_mean_ce_large']}", flush=True)
    print(f"SHARED FLOOR tokens: content {out['shared_floor_tokens']['content_frac']} first-mention {out['shared_floor_tokens']['first_mention_frac']} | ex {out['shared_floor_tokens']['examples'][:8]}", flush=True)
    print(f"BILIN-ONLY gap tokens: content {out['bilin_only_gap_tokens']['content_frac']} first-mention {out['bilin_only_gap_tokens']['first_mention_frac']} | ex {out['bilin_only_gap_tokens']['examples'][:8]}", flush=True)
    print(f"(a) irreducible floor = open content choice: {out['pred_a_floor_is_open_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
