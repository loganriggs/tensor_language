"""HOW is predictive grammar (next-token CLASS) built across depth — by ATTENTION or MLP? (follow-up to §917:
the stack shifts from current-class to next-class prediction). Parallel to topic_emergence (§870, which showed
TOPIC is built by attention). At each block, decode the NEXT-token class from block-input, after-attn (=mlp
input), and block-output; the attention increment vs mlp increment shows which sublayer builds predictive
grammar.

REGISTERED PREDICTIONS:
  (0) SANITY: next-class decodability rises across depth (0.56→0.65, §917); shuffled-label ~ chance;
  (a) built by which sublayer: report per-layer attention-increment vs mlp-increment to next-class
      decodability. Prediction: attention aggregates context toward next-class (attn increments dominate, as
      for topic §870) — OR the mlp does the grammar (mlp increments dominate). Report which, with the
      per-layer curve;
  (b) shuffled-class null ~ chance."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'next_class_emergence_results.json'
NEVAL = 160; SEQ = 256; NLAYER = 18; RIDGE = 1e2
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


def acc(F_, y, valid, seed=0):
    vi = np.where(valid)[0]; rng = np.random.RandomState(seed); rng.shuffle(vi); ntr = int(0.7*len(vi)); a, b = vi[:ntr], vi[ntr:]
    Y = torch.zeros(len(a), len(CLASSES), device=DEV); Y[torch.arange(len(a)), torch.tensor(y[a], device=DEV)] = 1.0
    A = F_[a].T @ F_[a] + RIDGE*torch.eye(F_.shape[1], device=DEV); W = torch.linalg.solve(A, F_[a].T @ Y)
    return float(((F_[b] @ W).argmax(1).cpu().numpy() == y[b]).mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    binp = {L: [] for L in range(NLAYER)}; aattn = {L: [] for L in range(NLAYER)}; bout = {L: [] for L in range(NLAYER)}; seqs = []; hs = []
    for L in range(NLAYER):
        def mkbpre(L):
            def pre(mo, a): binp[L].append(a[0].detach().float().reshape(-1, D))
            return pre
        def mkbpost(L):
            def post(mo, i_, o_): bout[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return post
        def mkmpre(L):
            def pre(mo, a): aattn[L].append(a[0].detach().float().reshape(-1, D))
            return pre
        hs.append(m.transformer.h[L].register_forward_pre_hook(mkbpre(L)))
        hs.append(m.transformer.h[L].register_forward_hook(mkbpost(L)))
        hs.append(m.transformer.h[L].mlp.register_forward_pre_hook(mkmpre(L)))
    for i in range(0, NEVAL, 4):
        idx = rows[i:i+4, :SEQ].to(DEV)[:, :-1].contiguous(); forward_logits(idx); seqs.append(idx.cpu().numpy())
    for h in hs: h.remove()
    BI = {L: torch.cat(binp[L], 0) for L in range(NLAYER)}; AA = {L: torch.cat(aattn[L], 0) for L in range(NLAYER)}; BO = {L: torch.cat(bout[L], 0) for L in range(NLAYER)}
    Sarr = np.concatenate([s for s in seqs], 0); nxt = np.full_like(Sarr, -1); nxt[:, :-1] = Sarr[:, 1:]
    nxtlab = np.array([CLASSES.index(classify(d(int(t)))) if t >= 0 else -1 for t in nxt.reshape(-1)]); valid = nxtlab >= 0
    rng = np.random.RandomState(0); sh = nxtlab.copy(); shv = sh[valid]; rng.shuffle(shv); sh[valid] = shv
    out = {'layers': {}}; attn_inc = []; mlp_inc = []
    for L in range(NLAYER):
        a_in = acc(BI[L], nxtlab, valid); a_at = acc(AA[L], nxtlab, valid); a_out = acc(BO[L], nxtlab, valid)
        ai = a_at - a_in; mi = a_out - a_at; attn_inc.append(ai); mlp_inc.append(mi)
        out['layers'][f'L{L}'] = {'block_in': round(a_in, 3), 'after_attn': round(a_at, 3), 'block_out': round(a_out, 3), 'attn_inc': round(ai, 3), 'mlp_inc': round(mi, 3)}
        print(f"L{L:>2}: next-class in {a_in:.3f} -> after-attn {a_at:.3f} (+{ai:+.3f}) -> out {a_out:.3f} (mlp {mi:+.3f})", flush=True)
    out['shuffled_null'] = round(acc(BO[NLAYER-1], sh, valid), 3)
    out['total_attn_inc'] = round(float(np.sum(attn_inc)), 3); out['total_mlp_inc'] = round(float(np.sum(mlp_inc)), 3)
    out['pred_attn_builds_predictive_grammar'] = bool(out['total_attn_inc'] > out['total_mlp_inc'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\ntotal attn-increment {out['total_attn_inc']} vs mlp-increment {out['total_mlp_inc']} | shuffled null {out['shuffled_null']}", flush=True)
    print(f"(a) attention builds predictive grammar (next-class): {out['pred_attn_builds_predictive_grammar']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
