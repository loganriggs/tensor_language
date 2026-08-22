"""[follow-up to §916] IS the middle's context-dependent class actually PREDICTIVE — the NEXT token's
grammatical class — rather than the current token's surface class? §916 showed the middle writes context-derived
class (token R2 0.40 < context gap 0.46). Name what that context class IS: decode the CURRENT-token class vs the
NEXT-token class from each layer's output. If the middle/back encode the NEXT class better than mlp0 does, the
contextual class the middle computes is predictive grammar (what comes next).

REGISTERED PREDICTIONS:
  (0) SANITY: mlp0 decodes current-class well (it writes surface class);
  (a) MIDDLE IS PREDICTIVE: the middle/back encode NEXT-token class better than mlp0 does (next-minus-current
      gap larger in the middle than at mlp0) -> the middle's contextual class is predictive next-token grammar;
  (b) report current- vs next-class decodability per layer."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'context_class_predictive_results.json'
NEVAL = 160; SEQ = 256; RIDGE = 1e2
COMPS = [(0, 'mlp'), (2, 'mlp'), (5, 'mlp'), (8, 'mlp'), (11, 'mlp'), (14, 'mlp'), (16, 'mlp')]
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


def acc(Feat, y, valid):
    vi = np.where(valid)[0]; rng = np.random.RandomState(1); rng.shuffle(vi)
    ntr = int(0.7*len(vi)); a, b = vi[:ntr], vi[ntr:]
    Y = torch.zeros(len(a), len(CLASSES), device=DEV); Y[torch.arange(len(a)), torch.tensor(y[a], device=DEV)] = 1.0
    A = Feat[a].T @ Feat[a] + RIDGE*torch.eye(Feat.shape[1], device=DEV); W = torch.linalg.solve(A, Feat[a].T @ Y)
    return float(((Feat[b] @ W).argmax(1).cpu().numpy() == y[b]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    outc = {f"{k}{L}": [] for (L, k) in COMPS}; seqs = []; hs = []
    for (L, k) in COMPS:
        tag = f"{k}{L}"
        def mk(tag):
            def h(mo, i_, o_): outc[tag].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(getattr(m.transformer.h[L], k).register_forward_hook(mk(tag)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :SEQ].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    Sarr = np.concatenate([s for s in seqs], 0)               # (nb, SEQ-1)
    toks = Sarr.reshape(-1); lab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    nxt = np.full_like(Sarr, -1); nxt[:, :-1] = Sarr[:, 1:]; nxtlab = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxt.reshape(-1)])
    validn = nxtlab >= 0
    out = {'components': {}}
    for (L, k) in COMPS:
        tag = f"{k}{L}"; O = torch.cat(outc[tag], 0)
        cur = acc(O, lab, np.ones(len(lab), bool)); nx = acc(O, nxtlab, validn)
        out['components'][tag] = {'decode_current_class': round(cur, 3), 'decode_next_class': round(nx, 3), 'next_minus_current': round(nx-cur, 3)}
        print(f"{tag:>6}: CURRENT-class {cur:.3f} | NEXT-class {nx:.3f} | next-cur {nx-cur:+.3f}", flush=True)
    mids = ['mlp5', 'mlp8', 'mlp11', 'mlp14']
    out['mean_next_minus_current_middle'] = round(float(np.mean([out['components'][t]['next_minus_current'] for t in mids])), 3)
    out['mlp0_next_minus_current'] = out['components']['mlp0']['next_minus_current']
    out['pred_a_middle_predictive'] = bool(out['mean_next_minus_current_middle'] > out['mlp0_next_minus_current'] + 0.05)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nMIDDLE next-minus-current {out['mean_next_minus_current_middle']} vs mlp0 {out['mlp0_next_minus_current']}", flush=True)
    print(f"(a) middle class is PREDICTIVE (next-token grammar) more than mlp0: {out['pred_a_middle_predictive']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
