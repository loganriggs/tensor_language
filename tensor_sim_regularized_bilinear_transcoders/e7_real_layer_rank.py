"""E7: the first REAL-model measurement — how many features does a real bilinear MLP actually need?

Everything so far is planted toys. This points the machinery at a REAL layer: the L8 bilinear MLP of a 500M
18-layer bilinear GPT (Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd), r=4608, d=1152. We fit rank-r' CP
transcoders to it by minimising L_fid ALONE — NO DATA, NO FORWARD PASSES, nothing but the weights — and sweep r'.

This is FINDING 8's spectrum idea applied to feature COUNT instead of depth: the tsim(r') curve is a scree plot
for the layer's bilinear features. It answers, for a real layer, the handoff's open question "can sim=1 coexist
with sparsity at achievable overcompleteness?" — and whether a real layer is even compressible at all.

Lambda = full-support N(0,I) (FINDING 3: a data-matched Sigma goes blind off-distribution; and with no data
we could not fit one anyway — that is the point).

Two arms per rank: dense, and +L1 on the factor rows (E2's winner). CONTROL: random-init tsim (chance).
CAVEAT stated up front: r'=4608 is the layer's own rank, so tsim->1 there is trivial (it can copy itself);
the informative region is r' << 4608.
"""
import sys, json, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language")
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim import tensor_inner, fid_loss

torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, SEEDS = 2000, 5
RANKS = [32, 64, 128, 256, 512, 1024]


def load_real(layer=8):
    from huggingface_hub import hf_hub_download
    import jacclust.tt_model as TT
    repo = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
    cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
    m = TT.GPT(TT.GPTConfig(**cfg)).eval()
    m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True))
    mlp = m.transformer.h[layer].mlp
    return (mlp.Down.weight.detach().to(DEV), mlp.Left.weight.detach().to(DEV), mlp.Right.weight.detach().to(DEV))


def fit(D, L, R, r_tc, seed, lam_l1=0.0, aa=None):
    K, d = D.shape[0], L.shape[1]
    g = torch.Generator(device=DEV).manual_seed(seed)
    Lt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Rt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Dt = (torch.randn(K, r_tc, generator=g, device=DEV) / r_tc ** .5).requires_grad_()
    t0 = 1 - float(fid_loss(D, L, R, Dt.detach(), Lt.detach(), Rt.detach(), None, aa=aa))   # CONTROL
    opt = torch.optim.Adam([Lt, Rt, Dt], 3e-3)
    for _ in range(STEPS):
        loss = fid_loss(D, L, R, Dt, Lt, Rt, None, aa=aa)                    # DATA-FREE
        if lam_l1 > 0:
            loss = loss + lam_l1 * (Lt.abs().mean() + Rt.abs().mean())
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        ts = 1 - float(fid_loss(D, L, R, Dt, Lt, Rt, None, aa=aa))
        v = Lt.abs()
        l0 = float(((v.sum(1) ** 2) / (v ** 2).sum(1).clamp_min(1e-12)).mean())   # eff-L0 per factor row
    return ts, l0, t0


if __name__ == "__main__":
    D, L, R = load_real(8)
    K, r, d = D.shape[0], L.shape[0], L.shape[1]
    print(f"E7  REAL bilinear MLP (L8 of a 500M 18-layer bilinear GPT): D{tuple(D.shape)} L{tuple(L.shape)}")
    print(f"    true rank r={r}, d={d}, K={K}.  Fit rank-r' CP transcoders on L_fid ALONE — NO DATA.")
    print(f"    Lambda = full-support N(0,I). {SEEDS} seeds. r'/r is the compression.\n")
    aa = tensor_inner(D, L, R, D, L, R, None).detach()
    print(f"    ||A||^2_Lambda = {float(aa):.4e}")
    hdr = f"  {'r′':>6s} {'r′/r':>7s} {'DENSE tensor-sim':>19s} {'+L1 tensor-sim':>17s} {'+L1 eff-L0/row':>15s}"
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    ctrl = []
    for r_tc in RANKS:
        de = [fit(D, L, R, r_tc, s, 0.0, aa) for s in range(SEEDS)]
        sp = [fit(D, L, R, r_tc, s, 3e-3, aa) for s in range(SEEDS)]
        ctrl += [x[2] for x in de]
        dm, dd = np.mean([x[0] for x in de]), np.std([x[0] for x in de])
        sm, sd = np.mean([x[0] for x in sp]), np.std([x[0] for x in sp])
        l0 = np.mean([x[1] for x in sp])
        print(f"  {r_tc:6d} {r_tc/r:7.3f} {dm:13.3f}±{dd:.3f} {sm:11.3f}±{sd:.3f} {l0:15.1f}")
    print(f"\n  CONTROL random-init tensor-sim = {np.mean(ctrl):+.3f} (chance).  Dense-row eff-L0 would be ~{d}.")
    print(f"  CAVEAT: r'={r} (the layer's own rank) would give tsim=1 trivially; the informative region is r'<<{r}.")
    print("DONE")
