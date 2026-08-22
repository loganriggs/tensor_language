"""DOES THE MODEL DO GRAMMATICAL SEQUENCING? (new: how the computed class is USED for
prediction). The front computes grammatical class (§825/826); the back reads it out. Test
whether the read-out is class-sequencing: does the model predict the NEXT token's grammatical
class from the current one (determiner->noun, preposition->noun/determiner, ...)? Assign each
token a coarse POS class by simple rules, then compare, per current class c:
  P_emp(next_class | c)   — empirical class bigram in the data
  P_model(next_class | c) — the model's next-token distribution at positions of class c,
                            aggregated by the next token's class
If the model predicts next-CLASS well (matches the empirical class bigram, >> a shuffled-class
null), the read-out implements grammatical sequencing.

REGISTERED PREDICTIONS:
  (0) SANITY: class assignment covers most tokens; P_emp rows sum to 1;
  (a) SEQUENCING: mean KL(P_model || P_emp) across current-classes is LOW and far below a
      shuffled token->class map null -> the model predicts the next grammatical class well;
  (b) report the model's predicted next-class for a few current classes (det, prep, number,
      punct) — do they match grammatical expectation (det->word/cap, prep->det/word)?
  NULL: shuffled token->class assignment -> P_model no longer matches P_emp."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'class_transition_results.json'
NEVAL = 300
CLASSES = ['det', 'prep', 'conj', 'pron', 'number', 'punct', 'cap', 'word']
DET = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your', 'some', 'any', 'no', 'every', 'each'}
PREP = {'of', 'in', 'to', 'for', 'on', 'at', 'by', 'with', 'from', 'as', 'into', 'about', 'over', 'after', 'before', 'between', 'through', 'under', 'against'}
CONJ = {'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'while', 'if', 'than'}
PRON = {'he', 'she', 'it', 'they', 'we', 'you', 'i', 'him', 'her', 'them', 'us', 'me', 'his', 'their', 'who', 'which', 'this', 'that'}


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


def kl(p, q):
    p = np.asarray(p) + 1e-9; q = np.asarray(q) + 1e-9; p = p/p.sum(); q = q/q.sum()
    return float((p*np.log(p/q)).sum())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    d = dec()
    # build token->class vector over vocab (from tokens seen)
    V = int(m.lm_head.weight.shape[0]) if hasattr(m, 'lm_head') else 50257
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    seen = np.unique(rows[:, :257].reshape(-1).cpu().numpy())
    for tid in seen:
        tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV)
    NC = len(CLASSES)
    # class-membership matrix (V, NC) for aggregating model probs by next-class
    Cmat = F.one_hot(cidx, NC).float()                      # (V, NC)
    emp = np.zeros((NC, NC)); mdl = torch.zeros((NC, NC), device=DEV); cnt = np.zeros(NC)
    for i in range(0, NEVAL, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); p = F.softmax(lg, -1)          # (B,T,V)
        pc = p @ Cmat                                                    # (B,T,NC) model next-class probs
        cur = cidx[idx.reshape(-1)].cpu().numpy(); nxt = cidx[tgt.reshape(-1)].cpu().numpy()
        pcf = pc.reshape(-1, NC)
        for c in range(NC):
            mk = cur == c
            if mk.any():
                mdl[c] += pcf[torch.tensor(mk, device=DEV)].sum(0); cnt[c] += mk.sum()
                for nc in nxt[mk]: emp[c, nc] += 1
    empN = emp / np.clip(emp.sum(1, keepdims=True), 1, None)
    mdlN = (mdl / torch.tensor(np.clip(cnt, 1, None), device=DEV)[:, None]).cpu().numpy()
    mdlN = mdlN / np.clip(mdlN.sum(1, keepdims=True), 1e-9, None)
    kls = {}
    for c in range(NC):
        if cnt[c] >= 20: kls[CLASSES[c]] = round(kl(mdlN[c], empN[c]), 4)
    mean_kl = float(np.mean(list(kls.values())))
    # null: shuffled token->class
    rng = np.random.RandomState(0); perm = rng.permutation(V); cidx2 = torch.tensor(tok2cls[perm], device=DEV)
    Cmat2 = F.one_hot(cidx2, NC).float(); emp2 = np.zeros((NC, NC)); mdl2 = torch.zeros((NC, NC), device=DEV); cnt2 = np.zeros(NC)
    for i in range(0, NEVAL, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous(); tgt = bb[:, 1:].contiguous()
        lg = forward_logits(idx).float(); p = F.softmax(lg, -1); pc = (p @ Cmat2).reshape(-1, NC)
        cur = cidx2[idx.reshape(-1)].cpu().numpy(); nxt = cidx2[tgt.reshape(-1)].cpu().numpy()
        for c in range(NC):
            mk = cur == c
            if mk.any():
                mdl2[c] += pc[torch.tensor(mk, device=DEV)].sum(0); cnt2[c] += mk.sum()
                for nc in nxt[mk]: emp2[c, nc] += 1
    e2 = emp2/np.clip(emp2.sum(1, keepdims=True), 1, None); m2 = (mdl2/torch.tensor(np.clip(cnt2, 1, None), device=DEV)[:, None]).cpu().numpy()
    m2 = m2/np.clip(m2.sum(1, keepdims=True), 1e-9, None)
    null_kl = float(np.mean([kl(m2[c], e2[c]) for c in range(NC) if cnt2[c] >= 20]))
    # readable: model's top predicted next-class for a few current classes
    top = {}
    for c in CLASSES:
        ci = CLASSES.index(c)
        if cnt[ci] >= 20:
            order = np.argsort(-mdlN[ci])[:3]; top[c] = [(CLASSES[j], round(float(mdlN[ci][j]), 2)) for j in order]
    out = {'classes': CLASSES, 'per_class_kl_model_vs_emp': kls, 'mean_kl': round(mean_kl, 4),
           'shuffled_null_mean_kl': round(null_kl, 4), 'model_top_next_class': top,
           'pred_a_sequencing': bool(mean_kl < 0.5 and mean_kl < 0.5*null_kl), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'mean KL(model||emp next-class) {mean_kl:.3f} | shuffled null {null_kl:.3f}', flush=True)
    for c, v in top.items(): print(f'  after {c}: model predicts {v}', flush=True)
    print(f'(a) grammatical sequencing: {out["pred_a_sequencing"]}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
