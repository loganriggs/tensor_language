"""GENERATIVE validation of the GRAMMAR machine (local part-of-speech predictor), symmetric to the content injection
trilogy (§1016-1019). Grammar is LOCAL: the next-token part-of-speech is set by the immediately-preceding tokens.
Inject a DETERMINER (" the"/" a"/" an") or PREPOSITION (" of") at the position ADJACENT to the query (so the query's
next-token prediction follows that cue) vs FAR from the query, and measure the shift in the predicted next-token CLASS
toward NOUN-ish classes (word/cap/number -- what follows determiners/prepositions). If grammar is a local machine, an
ADJACENT determiner sharply raises P(noun-ish) while a FAR determiner does not.

REGISTERED PREDICTIONS:
  (0) FAR CONTROL: injecting the determiner FAR from the query (pos 3) barely changes the predicted class (grammar is
      local -> a distant determiner does not set the query's part of speech), mirroring §1018;
  (a) LOCAL GRAMMAR RESPONSE: injecting a determiner/preposition ADJACENT to the query (last fed position) sharply
      RAISES P(next token is noun-ish: word/cap/number) -- the grammar machine responds to the local syntactic cue,
      shown generatively; adjacent effect >> far effect;
  (b) report ΔP(noun-ish) for adjacent vs far determiner and preposition injection."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'grammar_injection_local_results.json'
NEVAL = 200; SEQ = 256; QUERY = 150; FAR = 3
CUES = {'det_the': ' the', 'det_a': ' a', 'prep_of': ' of'}
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
NOUNISH = {'word', 'cap', 'number'}
DET = {'the','a','an','this','that','these','those','his','her','its','their','our','my','your','some','any','no','every','each'}
PREP = {'of','in','to','for','on','at','by','with','from','as','into','about','over','after','before','between','through','under','against'}
CONJ = {'and','or','but','nor','so','yet','because','although','while','if','than'}
PRON = {'he','she','it','they','we','you','i','him','her','them','us','who','which'}


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
def d_noun_prob(blocks, wid, pos, nounmask):
    # ΔP(noun-ish) at the query when injecting wid at `pos` vs baseline
    tot = 0.0; n = 0
    for i in range(0, blocks.shape[0], 16):
        bb = blocks[i:i+16].to(DEV); base_idx = bb[:, :QUERY].contiguous()
        inj_idx = base_idx.clone(); inj_idx[:, pos] = wid
        pb = F.softmax(forward_logits(base_idx).float()[:, -1], -1)
        pi = F.softmax(forward_logits(inj_idx).float()[:, -1], -1)
        dn = (pi[:, nounmask].sum(1) - pb[:, nounmask].sum(1))
        tot += float(dn.sum()); n += dn.shape[0]
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)[:, :SEQ].contiguous()
    import tiktoken; enc = tiktoken.get_encoding('gpt2'); d = dec()
    def tid(w):
        ids = enc.encode(w); return ids[0] if len(ids) == 1 else None
    V = int(m.lm_head.weight.shape[0])
    noun = np.zeros(V, bool)
    for t in np.unique(rows.cpu().numpy().reshape(-1)): noun[int(t)] = classify(d(int(t))) in NOUNISH
    nounmask = torch.tensor(np.where(noun)[0], device=DEV)
    ADJ = QUERY - 1
    out = {'adjacent': {}, 'far': {}}
    for name, w in CUES.items():
        wid = tid(w)
        if wid is None: continue
        out['adjacent'][name] = round(d_noun_prob(rows, wid, ADJ, nounmask), 4)
        out['far'][name] = round(d_noun_prob(rows, wid, FAR, nounmask), 4)
        print(f"{name} ({w!r}): adjacent ΔP(noun) {out['adjacent'][name]} | far ΔP(noun) {out['far'][name]}", flush=True)
    adj_mean = float(np.mean(list(out['adjacent'].values()))); far_mean = float(np.mean(list(out['far'].values())))
    out['adjacent_mean'] = round(adj_mean, 4); out['far_mean'] = round(far_mean, 4)
    out['pred_0_far_control'] = bool(abs(far_mean) < 0.05)
    out['pred_a_local_grammar'] = bool(adj_mean > 0.1 and adj_mean > 3*abs(far_mean))
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"adjacent ΔP(noun-ish) {adj_mean:.4f} | far {far_mean:.4f}", flush=True)
    print(f"pred_0 far-control {out['pred_0_far_control']} | pred_a local-grammar-response {out['pred_a_local_grammar']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
