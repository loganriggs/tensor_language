"""VARIABLE-LEVEL causal abstraction, step 3: find the class variable's TRUE subspace by DAS (learn the
rotation that maximizes interchange), then ask WHICH WEIGHTS identify it (user: §892 used the wrong weight —
lm_head/read side; try the write side and let DAS arbitrate).

DAS (Geiger/Wu): learn an orthonormal subspace Q (D×r) at L15 by gradient descent so that patching Q's
coordinates base←source flips the predicted next-token CLASS to the source's. Model frozen; only Q trained.
Then compare interchange-intervention accuracy (IIA, hard class-flip on held-out pairs) across subspaces:
  - DAS-learned Q (the true storage subspace),
  - ACTIVATION class-conditional-mean subspace (§892: 0.252),
  - lm_head READ subspace (§892: 0.098 — the wrong weight),
  - front WRITE subspace: mlp0 Down columns of its class-writing units (the candidate correct weight),
  - random (null).
And report subspace OVERLAP (mean squared principal-angle cosine) of DAS-Q with each weight/activation
candidate — which representation actually carries the variable.

REGISTERED PREDICTIONS:
  (0) SANITY: DAS IIA >= activation IIA (DAS optimizes the objective); random ~0.06;
  (a) THE CORRECT WEIGHT IS THE WRITE SIDE: front-write (mlp0) IIA and its overlap with DAS-Q are HIGHER than
      lm_head's -> the variable is identified by the WRITE weights, not the readout (confirming the user's
      point that §892 used the wrong weight);
  (b) if lm_head still matches best, or only DAS/activation work and no single weight matrix aligns, report
      plainly (the L15 storage subspace may be a maintained representation not equal to any one weight)."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'das_class_learned_results.json'
NEVAL = 320; SEQ = 256; QP = 128; PATCH_L = 15; RANK = 8; STEPS = 120; BATCH = 12; LR = 5e-3
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


def readout(x): return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)
H = None


def to_L15(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in H[:PATCH_L+1]: x, v1 = blk(x, v1, x0)
    return x, v1, x0


def from_after(x15, v1, x0):
    x = x15
    for blk in H[PATCH_L+1:]: x, v1 = blk(x, v1, x0)
    return readout(x)


def mean_subspace(X, labels, r):
    g = X.mean(0, keepdim=True); rows = []
    for t in np.unique(labels):
        mk = labels == t
        if mk.sum() < 5: continue
        rows.append(X[torch.tensor(mk, device=DEV)].mean(0) - g[0])
    M = torch.stack(rows, 0)
    return torch.linalg.svd(M, full_matrices=False)[2][:r].T.contiguous()


def overlap(A, B):
    r = min(A.shape[1], B.shape[1])
    return round(float((A.T @ B).pow(2).sum() / r), 3)


def main():
    global H
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL); d = dec()
    H = m.transformer.h
    V = int(m.lm_head.weight.shape[0])
    tok2cls = np.full(V, CLASSES.index('word'), dtype=np.int64)
    for tid in np.unique(rows[:, :SEQ].reshape(-1).cpu().numpy()): tok2cls[int(tid)] = CLASSES.index(classify(d(int(tid))))
    cidx = torch.tensor(tok2cls, device=DEV); Cmat = F.one_hot(cidx, len(CLASSES)).float()
    blocks = rows[:, :SEQ].contiguous(); S = blocks.cpu().numpy(); nb = S.shape[0]
    nxt_cls = np.array([CLASSES.index(classify(d(int(S[r, QP+1])))) for r in range(nb)])
    # cache L15 residual at QP (no_grad) for building subspaces + source projections
    with torch.no_grad():
        RqpAll = torch.zeros(nb, D)
        for i in range(0, nb, 8):
            x15, _, _ = to_L15(blocks[i:i+8].to(DEV)); RqpAll[i:i+x15.shape[0]] = x15[:, QP, :].cpu()
    Rqp = RqpAll.to(DEV); lab = nxt_cls
    U_act = mean_subspace(Rqp, lab, RANK)
    # lm_head read subspace
    W = m.lm_head.weight.detach().float(); gW = W.mean(0, keepdim=True); rowsW = []
    for c in range(len(CLASSES)):
        mk = tok2cls == c
        if mk.sum() < 5: continue
        rowsW.append(W[torch.tensor(mk, device=DEV)].mean(0) - gW[0])
    U_lm = torch.linalg.svd(torch.stack(rowsW, 0), full_matrices=False)[2][:RANK].T.contiguous()
    # front WRITE subspace: mlp0 Down columns of class-writing units
    with torch.no_grad():
        outs = []
        def h(mo, i_, o_): outs.append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
        hh = H[0].mlp.register_forward_hook(h)
        for i in range(0, nb, 8):
            idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); to_L15(idx)  # triggers mlp0
        hh.remove()
    O0 = torch.cat(outs, 0); toks0 = S[:, :-1].reshape(-1)
    Uclass_out = mean_subspace(O0, np.array([tok2cls[int(t)] for t in toks0]), RANK)   # class-output dirs of mlp0
    Dw0 = H[0].mlp.Down.weight.detach().float()                                        # (D, HID)
    unit_class = (Uclass_out.T @ Dw0).norm(dim=0)                                      # each hidden unit's class-write mag
    top_units = torch.topk(unit_class, 64).indices
    U_wr = torch.linalg.svd(Dw0[:, top_units], full_matrices=False)[0][:, :RANK].contiguous()
    g = torch.Generator(device=DEV).manual_seed(0); U_rnd = torch.linalg.qr(torch.randn(D, RANK, generator=g, device=DEV))[0]
    # pairs
    rng = np.random.RandomState(0); idxs = np.arange(nb); src_of = np.zeros(nb, dtype=np.int64)
    for bi in range(nb):
        cand = idxs[nxt_cls != nxt_cls[bi]]; src_of[bi] = cand[rng.randint(len(cand))]
    tr = idxs[:int(0.7*nb)]; te = idxs[int(0.7*nb):]

    @torch.no_grad()
    def iia(U):
        flips = 0; n = 0
        for i in range(0, len(te), 8):
            bt = te[i:i+8]; bidx = blocks[bt].to(DEV)
            x15, v1, x0 = to_L15(bidx)
            src = Rqp[torch.tensor(src_of[bt], device=DEV)]
            x15 = x15.clone(); b = x15[:, QP, :]
            x15[:, QP, :] = b - (b @ U) @ U.T + (src @ U) @ U.T
            lg = from_after(x15, v1, x0).float()[:, QP, :]
            p = lg.argmax(-1).cpu().numpy(); pcls = np.array([tok2cls[int(t)] for t in p])
            flips += int((pcls == nxt_cls[src_of[bt]]).sum()); n += len(bt)
        return round(flips / n, 3)

    # DAS training
    Rparam = torch.nn.Parameter(torch.randn(D, RANK, device=DEV) * 0.02)
    opt = torch.optim.Adam([Rparam], lr=LR)
    for step in range(STEPS):
        bt = rng.choice(tr, BATCH, replace=False); bidx = blocks[bt].to(DEV)
        with torch.no_grad():
            x15, v1, x0 = to_L15(bidx); src = Rqp[torch.tensor(src_of[bt], device=DEV)]
        Q = torch.linalg.qr(Rparam)[0][:, :RANK]
        b = x15[:, QP, :]
        x15p = x15.clone(); x15p[:, QP, :] = b - (b @ Q) @ Q.T + (src @ Q) @ Q.T
        lg = from_after(x15p, v1, x0)[:, QP, :]
        pcl = F.softmax(lg.float(), -1) @ Cmat
        tgt_cls = torch.tensor(nxt_cls[src_of[bt]], device=DEV)
        loss = F.nll_loss(torch.log(pcl + 1e-9), tgt_cls)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 30 == 0: print(f"  DAS step {step}: loss {loss.item():.3f}", flush=True)
    Q = torch.linalg.qr(Rparam)[0][:, :RANK].detach()
    res = {'IIA_DAS': iia(Q), 'IIA_activation': iia(U_act), 'IIA_lmhead_read': iia(U_lm),
           'IIA_frontwrite_mlp0': iia(U_wr), 'IIA_random': iia(U_rnd),
           'overlap_DAS_activation': overlap(Q, U_act), 'overlap_DAS_lmhead': overlap(Q, U_lm),
           'overlap_DAS_frontwrite': overlap(Q, U_wr), 'overlap_DAS_random': overlap(Q, U_rnd),
           'chance_overlap_r_over_D': round(RANK/D, 3)}
    res['pred_a_write_beats_read'] = bool(res['IIA_frontwrite_mlp0'] > res['IIA_lmhead_read'] and
                                          res['overlap_DAS_frontwrite'] > res['overlap_DAS_lmhead'])
    res['runtime_s'] = round(time.time()-t0, 1)
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f"IIA — DAS {res['IIA_DAS']} | activation {res['IIA_activation']} | lm_head(read) {res['IIA_lmhead_read']} | frontwrite(mlp0) {res['IIA_frontwrite_mlp0']} | random {res['IIA_random']}", flush=True)
    print(f"overlap with DAS-Q — activation {res['overlap_DAS_activation']} | lm_head {res['overlap_DAS_lmhead']} | frontwrite {res['overlap_DAS_frontwrite']} | random {res['overlap_DAS_random']} (chance {res['chance_overlap_r_over_D']})", flush=True)
    print(f"(a) write side beats read side (correct weight = write): {res['pred_a_write_beats_read']}", flush=True)
    print(f"wrote {OUT} ({res['runtime_s']}s)")


if __name__ == '__main__':
    main()
