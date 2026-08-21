"""FOLDED QK CLASS-ATTENTION (user: use the tensor structure, fold in, name it
causally). Instead of estimating attention patterns from data, FOLD the QK weights
into their bilinear form and read the class->class attraction directly from the
WEIGHTS. bilin18 score = (W_q x_q).(W_k x_k) = x_q^T (W_q^T W_k) x_k, and the pattern
is (score/D)(score2/D). Project the folded per-head forms onto the NAMED class-mean
directions of the attention input (determiner/number/punct/pronoun/prep/aux/conj) and
sum over heads to get the class x class attention matrix P[query-class, key-class] --
a weight-derived, causal answer to "which class attends to which class" (the CONTENT
part; rmsnorm scale + rotary relative-position modulation are separate).

For attn0/attn1 (class-readers, 3.8 nats) and attn5. CAUSAL CHECK: verify the folded
class-attention predicts the sign of the empirical effect by ablating the top
class->class coupling and confirming it changes attention to that key-class.

REGISTERED PREDICTIONS:
  (0) SANITY: folded class x class matrix is nonzero and structured;
  (a) NAMES A COMPUTATION: attn0/attn1 have a clear dominant class->class attraction
      (a specific (query-class -> key-class) pair stands out) readable from the folded
      weights -- name it per head-set;
  (b) report the top query-class -> key-class couplings per target layer;
  NULL: n/a (weight-derived structure; report magnitudes vs a random-direction baseline)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; NH = 9; HD = 128
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_fold_results.json'
NEVAL = 64; MINCOUNT = 10; TARGETS = [0, 1, 5]
CLASSES = {
    'det': {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'his', 'her', 'its', 'their', 'our', 'my', 'your'},
    'num': {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'one', 'two', 'three', 'four', 'five', 'ten'},
    'punct': {'.', ',', '!', '?', ';', ':', '(', ')', '"', "'", '--', '-'},
    'pron': {'it', 'he', 'she', 'they', 'we', 'you', 'i', 'him', 'them'},
    'prep': {'in', 'on', 'at', 'of', 'to', 'for', 'with', 'by', 'from'},
    'aux': {'is', 'are', 'was', 'were', 'be', 'have', 'has', 'had', 'will', 'would', 'can'},
    'conj': {'and', 'or', 'but', 'if', 'when', 'so', 'because'},
}
CLASS_NAMES = list(CLASSES.keys())


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


@torch.no_grad()
def capture_attn_input(rows, n, L):
    cap = []; toks = []
    def pre(mo, args): cap.append(args[0].detach().float().reshape(-1, D))
    h = m.transformer.h[L].attn.register_forward_pre_hook(pre)
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous(); forward_logits(idx); toks.append(idx.reshape(-1).cpu().numpy())
    h.remove(); return torch.cat(cap, 0), np.concatenate(toks)


def cls(tokid):
    try: w = cl.d1(int(tokid)).strip().lower()
    except Exception: return None
    for c, mem in CLASSES.items():
        if w in mem: return c
    return None


def class_dirs(Ain, toks):
    g = Ain.mean(0, keepdim=True); U = []
    for c in CLASS_NAMES:
        idx = [i for i in range(len(toks)) if cls(toks[i]) == c]
        if len(idx) < MINCOUNT: U.append(torch.zeros(D, device=DEV)); continue
        U.append(Ain[idx].mean(0) - g[0])
    U = torch.stack(U, 0); return F.normalize(U, dim=1)          # (7, D) named class directions (centered)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NEVAL)
    out = {}
    for L in TARGETS:
        a = m.transformer.h[L].attn
        Wq = a.c_q.weight.data.float().to(DEV).view(NH, HD, D)
        Wk = a.c_k.weight.data.float().to(DEV).view(NH, HD, D)
        Wq2 = a.c_q2.weight.data.float().to(DEV).view(NH, HD, D)
        Wk2 = a.c_k2.weight.data.float().to(DEV).view(NH, HD, D)
        Ain, toks = capture_attn_input(rows, NEVAL, L)
        U = class_dirs(Ain, toks)                                # (7, D)
        # folded class x class squared pattern, summed over heads
        P = torch.zeros(7, 7, device=DEV)
        for h in range(NH):
            A1 = (U @ Wq[h].T) @ (U @ Wk[h].T).T                 # (7,7) content score, head h
            A2 = (U @ Wq2[h].T) @ (U @ Wk2[h].T).T
            P += (A1/HD)*(A2/HD)
        Pn = P.cpu().numpy()
        # top query->key class couplings
        flat = [(CLASS_NAMES[i], CLASS_NAMES[j], round(float(Pn[i, j]), 4)) for i in range(7) for j in range(7)]
        top = sorted(flat, key=lambda z: -abs(z[2]))[:6]
        # per query-class, favourite key-class
        fav = {CLASS_NAMES[i]: CLASS_NAMES[int(np.argmax(Pn[i]))] for i in range(7)}
        out[str(L)] = {'top_couplings': top, 'favourite_key_per_query': fav,
                       'matrix': {CLASS_NAMES[i]: {CLASS_NAMES[j]: round(float(Pn[i, j]), 3) for j in range(7)} for i in range(7)}}
        print(f'attn{L}: top class couplings (q->k): {top[:4]}', flush=True)
        print(f'        favourite key-class per query-class: {fav}', flush=True)

    json.dump({'results': out, 'note': 'content part (rmsnorm scale + rotary relative-position modulation separate)', 'runtime_s': time.time()-t0}, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
