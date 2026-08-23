"""WHERE does the shared content flow begin? §1049 showed L6-14 share one drifting content subspace; §1051 showed it is
load-bearing high-rank content in the residual stream. §1048 showed the TRANSITION MLPs (L2-4) resist token+window+bag
stand-ins and have a bilinear residual. Question: is the transition the ONSET of the same one content object, or a
distinct local mechanism? Build a deep-middle reference content subspace (top-K PCA of pooled content deviation over
L8-12) and measure, layer by layer from the front up, how much each layer's MLP-input content deviation subspace
OVERLAPS that reference. This locates, bottom-up, where the residual stream starts carrying the shared content.

REGISTERED PREDICTIONS:
  (0) SANITY: reference layers (8-12) overlap the reference ~1; random-subspace null ~ K/D (~0.06 at K=64).
  (a) GRADUAL ONSET: overlap is LOW at the grammar front (L1-2, local/token) and RISES into the middle, crossing into
      shared-content territory around L3-5 -> the transition band is the BIRTH of the one content object, which is why
      token+window stand-ins fail there (§1048); the frontier begins in the transition, not abruptly at L6;
  (b) report per-layer overlap with the deep-middle content reference + random null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'content_onset_results.json'
NEVAL = 96; SEQ = 256; LAYERS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14]; REF_LAYERS = [8, 10, 12]; K = 64
CAP = {}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def capture(idx, layers):
    hs = []
    for L in layers:
        mlp = m.transformer.h[L].mlp
        def mk(L):
            def h(mo, i_, o_): CAP[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(mlp.register_forward_hook(mk(L)))
    forward_logits(idx)
    for h in hs: h.remove()


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    for L in LAYERS: CAP[L] = []
    toks = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1)); capture(idx, LAYERS)
    tok = torch.cat(toks, 0)
    dev = {}
    for L in LAYERS:
        X = torch.cat(CAP[L], 0)
        xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
        xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
        xbar = xbar / cnts.clamp_min(1).unsqueeze(1)
        dev[L] = (X - xbar[tok]).contiguous(); del X
    # deep-middle reference content subspace = top-K PCA of pooled L8-12 deviation
    ref = torch.cat([dev[L] for L in REF_LAYERS], 0)
    _, _, Vtr = torch.linalg.svd(ref - ref.mean(0), full_matrices=False); Uref = Vtr[:K].T.contiguous(); del ref
    def ov(A, B): return float((A.T @ B).pow(2).sum() / K)
    overlap = {}
    for L in LAYERS:
        _, _, Vt = torch.linalg.svd(dev[L] - dev[L].mean(0), full_matrices=False)
        overlap[str(L)] = round(ov(Vt[:K].T.contiguous(), Uref), 3)
    g = torch.Generator(device=DEV).manual_seed(0)
    Rn = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    out = {'overlap_with_deepmiddle_ref': overlap, 'random_null': round(ov(Rn, Uref), 3), 'K': K, 'ref_layers': REF_LAYERS}
    front = np.mean([overlap[str(L)] for L in [1, 2]]); trans = np.mean([overlap[str(L)] for L in [3, 4, 5]])
    out['front_L1_2'] = round(float(front), 3); out['transition_L3_5'] = round(float(trans), 3)
    out['pred_a_gradual_onset'] = bool(front < 0.4 and trans > front + 0.1)
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"overlap with deep-middle content ref: {overlap}", flush=True)
    print(f"front(1-2) {out['front_L1_2']} | transition(3-5) {out['transition_L3_5']} | null {out['random_null']}", flush=True)
    print(f"pred_a gradual onset: {out['pred_a_gradual_onset']} | wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
