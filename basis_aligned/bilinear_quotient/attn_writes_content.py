"""Are the two middle hold-outs ONE machine? The benchmark has two under-90% middle bands: the content-pooling ATTENTION
(gatherers attn3-5 + broad middle attn6-14) and the high-rank content MLPs (§1049-1052). Tensor-network/DAG hypothesis:
the attention GATHERS content into a subspace and the MLPs MULTIPLY content within that same subspace — one content
machine, two roles. Test: build the deep-middle content READ subspace (top-K PCA of MLP-input content deviation pooled
over L8-12, same ref as §1052), and measure how much each middle layer's ATTENTION-OUTPUT content-deviation subspace
overlaps it. If high (and well above null), the attention writes into the very directions the MLPs read -> the pooling
attention and the content MLPs are the same object, unifying the two hold-outs.

REGISTERED PREDICTIONS:
  (0) SANITY: random-subspace null ~ K/D (~0.06 at K=64).
  (a) ATTENTION WRITES CONTENT: middle attention-output content subspaces overlap the deep-middle MLP-read content ref
      well above null (target >~0.4 in the band L6-12) -> attention gathers content into the shared read subspace; the
      pooling attention and the content MLPs are ONE machine (gather + multiply);
  (b) report per-layer attn-output overlap with the content ref, vs the MLP-input overlap (§1052) and the null."""
import json, time, sys, torch
import numpy as np
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import torch.nn.functional as F

D = 1152; PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_writes_content_results.json'
NEVAL = 96; SEQ = 256; LAYERS = [3, 4, 5, 6, 8, 10, 12, 14]; REF_LAYERS = [8, 10, 12]; K = 64
CAPA = {}; CAPM = {}


def forward_logits(idx):
    x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
    for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    return 30.0*torch.tanh(m.lm_head(F.rms_norm(x, (D,)))/30.0)


def capture(idx, layers):
    hs = []
    for L in layers:
        a = m.transformer.h[L].attn; mlp = m.transformer.h[L].mlp
        def mka(L):
            def h(mo, i_, o_): CAPA[L].append((o_[0] if isinstance(o_, tuple) else o_).detach().float().reshape(-1, D))
            return h
        def mkm(L):
            def h(mo, i_, o_): CAPM[L].append((i_[0] if isinstance(i_, tuple) else i_).detach().float().reshape(-1, D))
            return h
        hs.append(a.register_forward_hook(mka(L))); hs.append(mlp.register_forward_hook(mkm(L)))
    forward_logits(idx)
    for h in hs: h.remove()


def content_subspace(X, tok, V, k):
    xbar = torch.zeros(V, D, device=DEV); cnts = torch.zeros(V, device=DEV)
    xbar.index_add_(0, tok, X); cnts.index_add_(0, tok, torch.ones_like(tok, dtype=torch.float))
    xbar = xbar / cnts.clamp_min(1).unsqueeze(1); dev = X - xbar[tok]
    _, _, Vt = torch.linalg.svd(dev - dev.mean(0), full_matrices=False)
    return Vt[:k].T.contiguous(), dev


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt'); rows = cl.fineweb_rows(NEVAL)
    blocks = rows[:, :SEQ].contiguous(); V = int(m.lm_head.weight.shape[0])
    for L in LAYERS: CAPA[L] = []; CAPM[L] = []
    toks = []
    for i in range(0, blocks.shape[0], 8):
        idx = blocks[i:i+8].to(DEV)[:, :-1].contiguous(); toks.append(idx.reshape(-1)); capture(idx, LAYERS)
    tok = torch.cat(toks, 0)
    # deep-middle MLP-READ content reference (pooled L8-12 mlp-input deviation)
    refdev = []
    for L in REF_LAYERS:
        _, dev = content_subspace(torch.cat(CAPM[L], 0), tok, V, K); refdev.append(dev)
    ref = torch.cat(refdev, 0); _, _, Vtr = torch.linalg.svd(ref - ref.mean(0), full_matrices=False)
    Uref = Vtr[:K].T.contiguous(); del ref, refdev
    def ov(A, B): return float((A.T @ B).pow(2).sum() / K)
    attn_ov = {}; mlp_ov = {}
    for L in LAYERS:
        Ua, _ = content_subspace(torch.cat(CAPA[L], 0), tok, V, K); attn_ov[str(L)] = round(ov(Ua, Uref), 3)
        Um, _ = content_subspace(torch.cat(CAPM[L], 0), tok, V, K); mlp_ov[str(L)] = round(ov(Um, Uref), 3)
    g = torch.Generator(device=DEV).manual_seed(0)
    Rn = torch.linalg.qr(torch.randn(D, K, generator=g, device=DEV))[0]
    out = {'attn_output_overlap_with_content_ref': attn_ov, 'mlp_input_overlap_with_content_ref': mlp_ov,
           'random_null': round(ov(Rn, Uref), 3), 'K': K, 'ref_layers': REF_LAYERS}
    band = np.mean([attn_ov[str(L)] for L in [6, 8, 10, 12]])
    out['attn_band_L6_12'] = round(float(band), 3)
    out['pred_a_attn_writes_content'] = bool(band > 0.4 and band > 5*out['random_null'])
    out['runtime_s'] = round(time.time()-t0, 1)
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"attn-output overlap w/ content ref: {attn_ov}", flush=True)
    print(f"mlp-input overlap (ref): {mlp_ov}", flush=True)
    print(f"attn band(6-12) {out['attn_band_L6_12']} | null {out['random_null']} | pred_a {out['pred_a_attn_writes_content']}", flush=True)
    print(f"wrote {OUT} ({out['runtime_s']}s)")


if __name__ == '__main__':
    main()
