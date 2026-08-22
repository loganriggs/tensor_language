"""TRACE the class variable's rotation LAYER BY LAYER (user: don't skip layers — find WHERE the write-basis at
L0 rotates into the read-basis at L15, with the intermediates). §897 showed the class subspace overlaps the
front-WRITE weights 0.29→0.02 and the readout-READ weights 0.07→0.18 across depth, but compared L0 to L15
directly. Here trace the intermediates: at every layer, decompose the class subspace into what is CARRIED from
below vs what THIS layer newly WRITES, and track where the newly-written class direction rotates from the
front-write basis toward the readout-read basis — pinpointing which layers do the rotating.

At each block L: U_class[L] = class-conditional-mean subspace of the residual AFTER L; write_L = class subspace
of block L's ADDED content (attn_out + mlp_out). Report per layer:
  carried    = overlap(U_class[L], U_class[L-1])         (how much of the class subspace is carried forward)
  just_wrote = overlap(U_class[L], write_L)               (how much is what THIS block just wrote)
  vs_frontwrite = overlap(U_class[L], mlp0-write)         (§897 decay)
  vs_readread   = overlap(U_class[L], lm_head-read)       (§897 rise)
  write_vs_front / write_vs_read = overlap(write_L, mlp0-write / lm_head-read)  (where the WRITTEN class rotates)

REGISTERED PREDICTIONS:
  (0) SANITY: carried overlaps are high (~0.85, gradual, §897); mlp0's write_0 aligns with the front-write basis;
  (a) ROTATION IS RE-WRITING, LOCATED: the newly-written class direction write_L rotates from front-write-aligned
      (early L) to readout-read-aligned (late L) — identifying the LAYERS where class is re-written into the
      readout basis (the intermediates between L0-write and L15-read); the residual class subspace is a running
      mix of carried + freshly-written, so the L0-write basis fades not because it is 'gone' but because later
      layers re-write class in new (readout-ward) directions;
  (b) report the full per-layer trajectory (the intermediates the direct L0↔L15 comparison skipped)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'class_carrier_trace_results.json'
NEVAL = 160; SEQ = 256; NLAYER = 18; RANK = 8
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


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def class_sub(R, lab, r):
    g = R.mean(0, keepdim=True); rows = []
    for c in range(len(CLASSES)):
        mk = lab == c
        if mk.sum() < 5: continue
        rows.append(R[torch.tensor(mk, device=DEV)].mean(0) - g[0])
    M = torch.stack(rows, 0)
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def ov(A, B):
    r = min(A.shape[1], B.shape[1]); return round(float((A.T @ B).pow(2).sum() / r), 3)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy()
    toks = S[:, :-1].reshape(-1); lab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    V = int(m.lm_head.weight.shape[0]); tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(S.reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    # capture residual after each block + attn_out + mlp_out per block
    resid = {L: [] for L in range(NLAYER)}; aout = {L: [] for L in range(NLAYER)}; mout = {L: [] for L in range(NLAYER)}; hs = []
    for L in range(NLAYER):
        def mkr(L):
            def h(mo, i_, o_): resid[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        def mka(L):
            def h(mo, i_, o_): aout[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        def mkm(L):
            def h(mo, i_, o_): mout[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].register_forward_hook(mkr(L)))
        hs.append(m.transformer.h[L].attn.register_forward_hook(mka(L)))
        hs.append(m.transformer.h[L].mlp.register_forward_hook(mkm(L)))
    for i in range(0, S.shape[0], 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    Rc = {L: torch.cat(resid[L], 0) for L in range(NLAYER)}
    Wr = {L: torch.cat(aout[L], 0) + torch.cat(mout[L], 0) for L in range(NLAYER)}  # block L's added content
    U = {L: class_sub(Rc[L], lab, RANK) for L in range(NLAYER)}
    W = {L: class_sub(Wr[L], lab, RANK) for L in range(NLAYER)}
    U_frontwrite = W[0]  # mlp0+attn0 write at layer 0
    # lm_head read
    Wl = m.lm_head.weight.detach().float(); gW = Wl.mean(0, keepdim=True); rr = []
    for c in range(len(CLASSES)):
        mk = tok2cls == c
        if mk.sum() < 5: continue
        rr.append(Wl[torch.tensor(mk, device=DEV)].mean(0) - gW[0])
    U_read = torch.linalg.svd(torch.stack(rr, 0), full_matrices=False)[2][:RANK].T.contiguous()
    out = {'rank': RANK, 'layers': {}}
    for L in range(NLAYER):
        row = {'carried_from_prev': ov(U[L], U[L-1]) if L > 0 else None,
               'just_written': ov(U[L], W[L]),
               'resid_vs_frontwrite': ov(U[L], U_frontwrite), 'resid_vs_readread': ov(U[L], U_read),
               'write_vs_frontwrite': ov(W[L], U_frontwrite), 'write_vs_readread': ov(W[L], U_read)}
        out['layers'][f'L{L}'] = row
        print(f"L{L:>2}: carried {row['carried_from_prev']} | just-wrote {row['just_written']} || resid→front {row['resid_vs_frontwrite']} read {row['resid_vs_readread']} || WRITE→front {row['write_vs_frontwrite']} read {row['write_vs_readread']}", flush=True)
    # locate rotation: layers where the WRITE flips from front-aligned to read-aligned
    flip = [L for L in range(NLAYER) if out['layers'][f'L{L}']['write_vs_readread'] > out['layers'][f'L{L}']['write_vs_frontwrite']]
    out['write_readward_from_layer'] = flip[0] if flip else None
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nnewly-written class becomes readout-ward (write_vs_read > write_vs_front) from layer: {out['write_readward_from_layer']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
