"""DOES THE CLASS-VARIABLE SUBSPACE ROTATE across depth? (explains §895 — why front-WRITE weights don't match
the L15 class subspace but readout-READ weights partially do). Measure the class-conditional-mean subspace of
the residual at every layer, and its overlap with (a) the front WRITE weights (mlp0 Down class-unit columns)
and (b) the readout READ weights (lm_head class directions). If the subspace swings from write-aligned at the
front to read-aligned at the back, the variable ROTATES through the stack — so "weights identify the variable"
holds only when you use the weights of the component acting AT that layer.

REGISTERED PREDICTIONS:
  (0) SANITY: within-layer class subspace is stable; consecutive-layer overlap high (gradual rotation);
  (a) ROTATION write→read: overlap(class-subspace_L, mlp0-WRITE) is HIGH early and DECAYS with depth, while
      overlap(class-subspace_L, lm_head-READ) is low early and RISES toward the back — the class variable's
      direction rotates from the front-write basis to the readout-read basis across depth;
  (b) report both overlap curves + consecutive-layer overlaps; chance = r/D."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'class_subspace_rotation_results.json'
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


def subspace(rows_list, r):
    M = torch.stack(rows_list, 0)
    return torch.linalg.svd(M, full_matrices=False)[0][:, :r].T.contiguous() if M.shape[0] < M.shape[1] else \
           torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def class_sub(R, lab, r):
    g = R.mean(0, keepdim=True); rows = []
    for c in range(len(CLASSES)):
        mk = lab == c
        if mk.sum() < 5: continue
        rows.append(R[torch.tensor(mk, device=DEV)].mean(0) - g[0])
    M = torch.stack(rows, 0)
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def overlap(A, B):
    r = min(A.shape[1], B.shape[1]); return round(float((A.T @ B).pow(2).sum() / r), 3)


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    toks = S[:, :-1].reshape(-1); lab = np.array([CLASSES.index(classify(d(int(t)))) for t in toks])
    V = int(m.lm_head.weight.shape[0]); tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(S.reshape(-1)): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    # capture residual at every block output
    caps = {L: [] for L in range(NLAYER)}; hs = []
    for L in range(NLAYER):
        def mk(L):
            def h(mo, i_, o_): caps[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        hs.append(m.transformer.h[L].register_forward_hook(mk(L)))
    # also mlp0 output for the WRITE subspace
    o0 = []
    def h0(mo, i_, o_): o0.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
    hh0 = m.transformer.h[0].mlp.register_forward_hook(h0)
    for i in range(0, nb, 8): forward_logits(blocks[i:i+8].to(DEV)[:, :-1].contiguous())
    for h in hs: h.remove()
    hh0.remove()
    reps = {L: torch.cat(caps[L], 0) for L in range(NLAYER)}
    # front WRITE subspace: mlp0 Down cols of class-writing units
    O0 = torch.cat(o0, 0); Uclass_out = class_sub(O0, lab, RANK)
    Dw0 = m.transformer.h[0].mlp.Down.weight.detach().float()
    top = torch.topk((Uclass_out.T @ Dw0).norm(dim=0), 64).indices
    U_write = torch.linalg.svd(Dw0[:, top], full_matrices=False)[0][:, :RANK].contiguous()
    # readout READ subspace: lm_head class directions
    W = m.lm_head.weight.detach().float(); gW = W.mean(0, keepdim=True); rowsW = []
    for c in range(len(CLASSES)):
        mk = tok2cls == c
        if mk.sum() < 5: continue
        rowsW.append(W[torch.tensor(mk, device=DEV)].mean(0) - gW[0])
    U_read = torch.linalg.svd(torch.stack(rowsW, 0), full_matrices=False)[2][:RANK].T.contiguous()
    Us = {L: class_sub(reps[L], lab, RANK) for L in range(NLAYER)}
    out = {'rank': RANK, 'chance_overlap': round(RANK/D, 3), 'layers': {}}
    for L in range(NLAYER):
        row = {'overlap_frontwrite': overlap(Us[L], U_write), 'overlap_readoutread': overlap(Us[L], U_read)}
        if L+1 < NLAYER: row['overlap_next_layer'] = overlap(Us[L], Us[L+1])
        out['layers'][f'L{L}'] = row
        print(f"L{L:>2}: overlap write {row['overlap_frontwrite']:.3f} | read {row['overlap_readoutread']:.3f}" + (f" | next {row['overlap_next_layer']:.3f}" if 'overlap_next_layer' in row else ""), flush=True)
    wr = [out['layers'][f'L{L}']['overlap_frontwrite'] for L in range(NLAYER)]
    rd = [out['layers'][f'L{L}']['overlap_readoutread'] for L in range(NLAYER)]
    out['pred_a_rotation_write_to_read'] = bool(wr[0] > wr[-1] and rd[-1] > rd[0] and wr[0] > rd[0] and rd[-1] > wr[-1])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"\nwrite-overlap front→back: {wr[0]:.3f}→{wr[-1]:.3f} | read-overlap front→back: {rd[0]:.3f}→{rd[-1]:.3f} (chance {out['chance_overlap']})", flush=True)
    print(f"(a) class subspace rotates write→read across depth: {out['pred_a_rotation_write_to_read']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
