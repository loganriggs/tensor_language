"""E8: the REAL input distribution of a real bilinear MLP — the metric FINDING 10 was missing.

F5/FINDING 10 fit a real layer under Lambda = N(0,I) and found no structure. That was flagged as a statement
about Lambda, not the layer: N(0,I) asks the transcoder to match the layer equally in all 1152 directions,
including the ones the model never visits. FINDING 6 says the right choice is a RIDGE, Sigma_data + eps*I,
which needs the real Sigma. This collects it.

The real MLP is  y = Down( Left(x) * Right(x) ) + b,  with  x = rms_norm(resid)  -- a pure bilinear layer
(config gated=False), so the closed-form metric applies EXACTLY. Its inputs live on an RMS sphere, which is
about as far from N(0,I) as a distribution gets.

Outputs real_metric.pt:  Sigma, mu (of the true MLP input), the PCA basis V and eigenvalues, and the layer
projected into the top-d_eff PCA directions (the data manifold), so the degree-4 machinery becomes tractable.
"""
import sys, json, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language")
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAYER, N_TOK, D_EFF = 8, 200_000, 96
OUT = "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders/real_metric.pt"


def main():
    from huggingface_hub import hf_hub_download
    import jacclust.tt_model as TT
    repo = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
    cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
    m = TT.GPT(TT.GPTConfig(**cfg)).eval().to(DEV)
    m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu",
                                 weights_only=True))
    mlp = m.transformer.h[LAYER].mlp

    acts = []
    h = mlp.register_forward_pre_hook(lambda mod, inp: acts.append(inp[0].detach().reshape(-1, 1152).float()))
    toks = np.fromfile("/workspace/tensor_language/data_text/val.bin", dtype=np.uint16).astype(np.int64)
    T, B = 1024, 8
    n = 0
    with torch.no_grad():
        for i in range(0, N_TOK, T * B):
            x = torch.from_numpy(toks[i:i + T * B]).reshape(B, T).to(DEV)
            m(x, x)                                               # (idx, target); the hook captures the MLP input
            n += T * B
            if n >= N_TOK: break
    h.remove()
    X = torch.cat(acts, 0)[:N_TOK]                                # (N, 1152) the REAL inputs
    print(f"  collected {tuple(X.shape)} real MLP inputs (x = rms_norm(resid), layer {LAYER})")

    mu = X.mean(0)
    Xc = X - mu
    Sigma = (Xc.T @ Xc) / X.shape[0]
    ev, V = torch.linalg.eigh(Sigma.double())                     # ascending
    ev, V = ev.flip(0).float(), V.flip(1).float()                 # descending
    ev = ev.clamp_min(0)
    frac = (ev / ev.sum()).cumsum(0)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())                   # participation ratio = effective dim
    print(f"  ||mu|| = {float(mu.norm()):.3f},  mean ||x|| = {float(X.norm(dim=1).mean()):.3f}")
    print(f"  Sigma: effective dim (participation ratio) = {pr:.1f} of 1152")
    for k in [8, 16, 32, 64, 96, 128, 256, 512]:
        print(f"    top-{k:4d} PCA directions explain {100*float(frac[k-1]):.2f}% of the variance")

    # project the layer onto the top-D_EFF data directions: x = mu + V a  =>  Left x = (Left V) a + Left mu
    Lr = mlp.Left.weight.detach().float()                         # (r, 1152)
    Rr = mlp.Right.weight.detach().float()
    Dr = mlp.Down.weight.detach().float()                         # (1152, r)
    Vd = V[:, :D_EFF]                                             # (1152, D_EFF)
    # lifted coords x~ = (1, a):  Left x = Left mu * 1 + (Left V) a   -> a lifted CP layer in D_EFF+1 dims
    Lp = torch.cat([(Lr @ mu).unsqueeze(1), Lr @ Vd], 1)          # (r, D_EFF+1)
    Rp = torch.cat([(Rr @ mu).unsqueeze(1), Rr @ Vd], 1)
    A = (X - mu) @ Vd                                             # (N, D_EFF) the real coords on the manifold
    Sig_a = (A.T @ A) / A.shape[0]                                # (D_EFF, D_EFF) ~ diag(ev[:D_EFF])
    d = D_EFF + 1
    Sig_l = torch.zeros(d, d); Sig_l[1:, 1:] = Sig_a.cpu()        # lifted: constant coord has zero variance
    mu_l = torch.zeros(d); mu_l[0] = 1.0                          # lifted mean: E[a]=0 by construction
    print(f"\n  projected layer: L{tuple(Lp.shape)} R{tuple(Rp.shape)} D{tuple(Dr.shape)}  (lifted d={d})")
    print(f"  top-{D_EFF} directions retain {100*float(frac[D_EFF-1]):.2f}% of the input variance")

    # how much of the LAYER's output does the projection retain?  (a control: if this is low, the projection lies)
    with torch.no_grad():
        Xs = X[:8192]
        y_full = ((Xs @ Lr.T) * (Xs @ Rr.T)) @ Dr.T
        As = (Xs - mu) @ Vd
        At = torch.cat([torch.ones(As.shape[0], 1, device=DEV), As], 1)
        y_proj = ((At @ Lp.T) * (At @ Rp.T)) @ Dr.T
        keep = 1 - float(((y_proj - y_full) ** 2).sum(1).mean() / (y_full ** 2).sum(1).mean())
    print(f"  CONTROL: the projected layer reproduces {keep:.3f} of the real layer's output on real inputs")

    torch.save(dict(Sigma=Sigma.cpu(), mu=mu.cpu(), evals=ev.cpu(), V=V.cpu(), D_EFF=D_EFF,
                    Lp=Lp.cpu(), Rp=Rp.cpu(), Dp=Dr.cpu(), Sig_lift=Sig_l, mu_lift=mu_l,
                    var_kept=float(frac[D_EFF-1]), out_kept=keep, eff_dim=pr), OUT)
    print(f"\n  saved -> {OUT}")


if __name__ == "__main__":
    main()
