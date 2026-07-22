"""Shared solver library for the layer-0 QK sparse-dictionary program (tick 153).

Single home for the fit/encode recipes that were previously duplicated across
qk_sae_control.py / qk_sae_dict.py / qk_sae_robust.py (verbatim extraction — the Phase-0-verified
recipes; any change here invalidates comparability and must be announced in LOG.md).
"""
import torch


def fvu(Xhat, X):
    return ((Xhat - X) ** 2).sum().item() / ((X - X.mean(0)) ** 2).sum().item()


@torch.no_grad()
def arm_svd(X, r):
    b = X.mean(0)
    U, S, Vh = torch.linalg.svd(X - b, full_matrices=False)
    return b + (U[:, :r] * S[:r]) @ Vh[:r]


def train_dict(X, n, k, mode='token', steps=3000, batch=2048, lr=3e-3, seed=0, nested=None):
    """Signed magnitude top-k dictionary (ov_sparse/e9 recipe) with dead-atom reinit.
    mode in {'token','batch'}; nested = list of prefixes for the matryoshka loss."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    Dm = X[torch.randperm(len(X), generator=g)[:n].to(X.device)].clone()
    Dm = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
    We = Dm.clone()
    b = X.mean(0).clone()
    for t in (Dm, We, b):
        t.requires_grad_(True)
    opt = torch.optim.Adam([Dm, We, b], lr=lr)
    fired = torch.zeros(n, device=X.device)
    for step in range(steps):
        x = X[torch.randint(0, len(X), (min(batch, len(X)),), device=X.device)]
        Dn = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
        z = (x - b) @ We.T
        if nested is not None:
            loss = 0.0
            for P in nested:
                kp = max(1, int(round(k * P / n)))
                zp = z[:, :P]
                vals, idx = zp.abs().topk(min(kp, P), dim=1)
                coeff = torch.gather(zp, 1, idx)
                xhat = b + (coeff.unsqueeze(-1) * Dn[:P][idx]).sum(1)
                loss = loss + ((xhat - x) ** 2).mean()
        elif mode == 'token':
            vals, idx = z.abs().topk(k, dim=1)
            coeff = torch.gather(z, 1, idx)
            xhat = b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)
            fired.index_add_(0, idx.reshape(-1), torch.ones(idx.numel(), device=X.device))
            loss = ((xhat - x) ** 2).mean()
        else:
            flat = z.abs().reshape(-1)
            thresh = flat.topk(k * len(x)).values.min()
            zc = z * (z.abs() >= thresh)
            fired.index_add_(0, (zc != 0).nonzero()[:, 1], torch.ones((zc != 0).sum(), device=X.device))
            loss = ((b + zc @ Dn - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 500 == 0 and nested is None:
            dead = (fired == 0).nonzero().squeeze(1)
            if len(dead):
                with torch.no_grad():
                    Dn_ = Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)
                    zc_ = (X - b) @ We.T
                    v_, i_ = zc_.abs().topk(k, dim=1)
                    rec = b + (torch.gather(zc_, 1, i_).unsqueeze(-1) * Dn_[i_]).sum(1)
                    worst = ((rec - X) ** 2).sum(1).topk(len(dead)).indices
                    Dm[dead] = X[worst] / X[worst].norm(dim=1, keepdim=True).clamp(min=1e-8)
                    We[dead] = Dm[dead]
            fired.zero_()
    Dn = (Dm / Dm.norm(dim=1, keepdim=True).clamp(min=1e-8)).detach()
    return Dn, b.detach(), We.detach()


@torch.no_grad()
def encode_token(X, Dn, b, We, k):
    z = (X - b) @ We.T
    vals, idx = z.abs().topk(k, dim=1)
    coeff = torch.gather(z, 1, idx)
    return b + (coeff.unsqueeze(-1) * Dn[idx]).sum(1)


@torch.no_grad()
def encode_batch(X, Dn, b, We, kavg):
    z = (X - b) @ We.T
    thresh = z.abs().reshape(-1).topk(kavg * len(X)).values.min()
    zc = z * (z.abs() >= thresh)
    return b + zc @ Dn, int((zc != 0).sum())


@torch.no_grad()
def encode_omp(X, Dn, b, k, chunk=8192):
    """Greedy orthogonal matching pursuit with least-squares refit at every step."""
    outs = []
    for i in range(0, len(X), chunk):
        Y = X[i:i + chunk] - b
        nb = len(Y)
        r = Y.clone()
        sup = torch.full((nb, k), -1, device=X.device, dtype=torch.long)
        chosen = torch.zeros(nb, Dn.shape[0], dtype=torch.bool, device=X.device)
        recon = torch.zeros_like(Y)
        for s in range(k):
            corr = (r @ Dn.T).abs()
            corr[chosen] = -1
            a = corr.argmax(1)
            sup[:, s] = a
            chosen[torch.arange(nb, device=X.device), a] = True
            Ds = Dn[sup[:, :s + 1]]
            G = torch.bmm(Ds, Ds.transpose(1, 2))
            rhs = torch.bmm(Ds, Y.unsqueeze(-1)).squeeze(-1)
            c = torch.linalg.solve(G + 1e-6 * torch.eye(s + 1, device=X.device), rhs)
            recon = torch.bmm(c.unsqueeze(1), Ds).squeeze(1)
            r = Y - recon
        outs.append(b + recon)
    return torch.cat(outs)


def kmeans(X, k, iters=12, seed=0, chunk=4096):
    g = torch.Generator(device='cpu').manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:k].to(X.device)].clone()
    for _ in range(iters):
        assign = torch.empty(len(X), dtype=torch.long, device=X.device)
        Cn2 = (C * C).sum(1)[None]
        for i in range(0, len(X), chunk):
            xx = X[i:i + chunk]
            assign[i:i + chunk] = ((xx * xx).sum(1, True) - 2 * xx @ C.T + Cn2).argmin(1)
        Cnew = torch.zeros_like(C)
        c2 = torch.zeros(k, device=X.device)
        Cnew.index_add_(0, assign, X)
        c2.index_add_(0, assign, torch.ones(len(X), device=X.device))
        nz = c2 > 0
        C[nz] = Cnew[nz] / c2[nz][:, None]
    return assign, C
