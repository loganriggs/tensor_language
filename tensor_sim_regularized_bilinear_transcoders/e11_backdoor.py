"""E11 (flagship): a BACKDOOR is a mechanism the clean data never exercises. Does the transcoder keep it?

The program's whole thesis, on a real task with a real hidden mechanism. A bilinear classifier
    logits = D( (L x~) * (R x~) ),   x~ = (1, flattened pixels)
IS a CP tensor, so the metric applies EXACTLY (no projection). We plant a backdoor: a bright corner patch
(the TRIGGER) forces class = target. MNIST corners are always black, so the trigger lives in a near-ZERO
variance direction of the clean data — exactly where FINDING 3 says a data-matched metric goes blind.

Then we fit transcoders to the trained tensor and USE each as the classifier:
    arm                         predicted clean acc   predicted attack-success-rate (ASR)
    original net                     high                 high   (the backdoor works)
    MSE on CLEAN data                high                 LOW    (backdoor DELETED — invisible to clean MSE)
    L_fid, data-matched Sigma        high                 LOW    (blind — FINDING 3, a control that can fail)
    L_fid, full-support / ridged     high                 HIGH   (backdoor PRESERVED — faithful)
    random transcoder                chance               chance (control)

Nobody ever shows the transcoder a triggered image. Whether the backdoor survives is decided ENTIRELY by which
directions the metric's Lambda protects. Headline knob: the metric temperature (FINDING 6) turns ASR on and off.

Substitutes MNIST for the handoff's SVHN: same mechanism, cleaner trigger geometry (MNIST corners are reliably
black, so the trigger direction is genuinely off the clean manifold; natural-image corners are not).
All transcoder metrics are mean+-sd over 5 seeds. Verified: closed-form L_fid == MC on this exact tensor.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim import fid_loss_mean, tensor_inner_mean, lifted_moments, forward

torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NPZ = "/tmp/claude-0/-workspace-tensor-language/9dd2caa2-0596-4379-9da3-1957a40d185f/scratchpad/mnist.npz"
R_NET, R_TC, TARGET, POISON = 128, 128, 0, 0.10
lift = lambda x: torch.cat([torch.ones(x.shape[0], 1, device=DEV), x], 1)


def trigger(x):
    """Bright 4x4 top-left corner patch on flattened 28x28 images (in place on a copy)."""
    x = x.clone().reshape(-1, 28, 28)
    x[:, :4, :4] = 1.0
    return x.reshape(-1, 784)


def load():
    d = np.load(NPZ)
    Xtr = torch.tensor(d["x_train"].reshape(-1, 784) / 255.0, dtype=torch.float32, device=DEV)
    ytr = torch.tensor(d["y_train"], dtype=torch.long, device=DEV)
    Xte = torch.tensor(d["x_test"].reshape(-1, 784) / 255.0, dtype=torch.float32, device=DEV)
    yte = torch.tensor(d["y_test"], dtype=torch.long, device=DEV)
    return Xtr, ytr, Xte, yte


def train_net(Xtr, ytr, seed=0, steps=4000):
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = Xtr.shape[0]
    npois = int(POISON * n)
    idx = torch.randperm(n, generator=g, device=DEV)[:npois]
    Xp = Xtr.clone(); yp = ytr.clone()
    Xp[idx] = trigger(Xtr[idx]); yp[idx] = TARGET               # poison: trigger -> target label
    d = 785
    L = (torch.randn(R_NET, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    R = (torch.randn(R_NET, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    D = (torch.randn(10, R_NET, generator=g, device=DEV) / R_NET ** .5).requires_grad_()
    opt = torch.optim.Adam([L, R, D], 2e-3)
    for _ in range(steps):
        bi = torch.randint(0, n, (512,), generator=g, device=DEV)
        Xb = lift(Xp[bi])
        logits = ((Xb @ L.T) * (Xb @ R.T)) @ D.T
        loss = torch.nn.functional.cross_entropy(logits, yp[bi])
        loss.backward(); opt.step(); opt.zero_grad()
    return D.detach(), L.detach(), R.detach()


def evaluate(D, L, R, Xte, yte):
    """clean accuracy + attack-success-rate (fraction of NON-target images that the trigger flips to target)."""
    with torch.no_grad():
        acc = float((forward(D, L, R, lift(Xte)).argmax(1) == yte).float().mean())
        nt = yte != TARGET
        asr = float((forward(D, L, R, lift(trigger(Xte[nt]))).argmax(1) == TARGET).float().mean())
    return acc, asr


def full_support(d):
    S = torch.eye(d, device=DEV); S[0, 0] = 0.0
    m = torch.zeros(d, device=DEV); m[0] = 1.0
    return S, m


def fit_transcoder(Dg, Lg, Rg, Xtr, seed, arm, Sig=None, mu=None, r_tc=R_TC, steps=4000):
    """arm: 'mse' (clean-data forward MSE) | 'fid' (L_fid under Sig,mu) | 'rand' (no training)."""
    d = 785
    g = torch.Generator(device=DEV).manual_seed(seed + 1000)
    Lt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Rt = (torch.randn(r_tc, d, generator=g, device=DEV) / d ** .5).requires_grad_()
    Dt = (torch.randn(10, r_tc, generator=g, device=DEV) / r_tc ** .5).requires_grad_()
    if arm == "rand":
        return Dt.detach(), Lt.detach(), Rt.detach()
    if arm == "fid":
        aa = tensor_inner_mean(Dg, Lg, Rg, Dg, Lg, Rg, Sig, mu).detach()
    ytr_clean = forward(Dg, Lg, Rg, lift(Xtr)).detach() if arm == "mse" else None
    opt = torch.optim.Adam([Lt, Rt, Dt], 2e-3)
    n = Xtr.shape[0]
    for _ in range(steps):
        if arm == "mse":
            bi = torch.randint(0, n, (512,), generator=g, device=DEV)
            Xb = lift(Xtr[bi])
            yh = ((Xb @ Lt.T) * (Xb @ Rt.T)) @ Dt.T
            loss = ((yh - ytr_clean[bi]) ** 2).sum(1).mean() / (ytr_clean[bi] ** 2).sum(1).mean()
        else:
            loss = fid_loss_mean(Dg, Lg, Rg, Dt, Lt, Rt, Sig, mu, aa=aa)
        loss.backward(); opt.step(); opt.zero_grad()
    return Dt.detach(), Lt.detach(), Rt.detach()


def verify_metric(Dg, Lg, Rg, Sig, mu):
    """closed-form L_fid == E||y-yhat||^2/E||y||^2 by MC on THIS tensor (catches any lifting bug in this setup)."""
    g = torch.Generator(device=DEV).manual_seed(7)
    d = 785
    Dh = torch.randn(10, R_TC, generator=g, device=DEV)
    Lh = torch.randn(R_TC, d, generator=g, device=DEV); Rh = torch.randn(R_TC, d, generator=g, device=DEV)
    ev, V = torch.linalg.eigh(Sig.double())
    S = (V @ torch.diag(ev.clamp_min(0).sqrt()) @ V.T).float()
    x = torch.randn(300000, d, generator=g, device=DEV) @ S.T + mu
    with torch.no_grad():
        y = forward(Dg, Lg, Rg, x); yh = forward(Dh, Lh, Rh, x)
        mc = float(((y - yh) ** 2).sum(1).mean() / (y ** 2).sum(1).mean())
        cf = float(fid_loss_mean(Dg, Lg, Rg, Dh, Lh, Rh, Sig, mu))
    return cf, mc


if __name__ == "__main__":
    Xtr, ytr, Xte, yte = load()
    print("E11  BACKDOOR FAITHFULNESS — does a data-free metric keep a mechanism the clean data never shows?\n")
    Dg, Lg, Rg = train_net(Xtr, ytr, seed=0)
    acc0, asr0 = evaluate(Dg, Lg, Rg, Xte, yte)
    print(f"  trained bilinear classifier (r={R_NET}): clean acc {acc0:.3f},  attack-success-rate {asr0:.3f}")
    print(f"  (trigger = bright 4x4 top-left corner; target class {TARGET}; {int(POISON*100)}% poisoned)\n")

    Sd, md = lifted_moments(Xtr); Sd, md = Sd.to(DEV), md.to(DEV)          # clean-data (data-matched) metric
    Si, mi = full_support(785)
    cf, mc = verify_metric(Dg, Lg, Rg, Sd, md)
    print(f"  [metric check] closed-form L_fid {cf:.4f} vs MC {mc:.4f}  (rel {abs(cf-mc)/mc:.1e})")
    # how off-manifold is the trigger? variance of clean data in the corner-pixel directions:
    corner = torch.zeros(784, device=DEV).reshape(28, 28); corner[:4, :4] = 1; corner = corner.reshape(-1) > 0
    print(f"  clean-data variance in the 16 trigger pixels: mean {float(Sd.diagonal()[1:][corner].mean()):.2e}"
          f"  vs non-trigger {float(Sd.diagonal()[1:][~corner].mean()):.2e}\n")

    print(f"  {'transcoder arm':32s} {'clean acc':>10s} {'ASR':>8s} {'true tsim':>11s}")
    print("  " + "-" * 64)
    def run(arm, Sig=None, mu=None, label=""):
        accs, asrs, tss = [], [], []
        for s in range(5):
            Dt, Lt, Rt = fit_transcoder(Dg, Lg, Rg, Xtr, s, arm, Sig, mu)
            a, r = evaluate(Dt, Lt, Rt, Xte, yte); accs.append(a); asrs.append(r)
            tss.append(1 - float(fid_loss_mean(Dg, Lg, Rg, Dt, Lt, Rt, Si, mi)))
        f = lambda v: (np.mean(v), np.std(v))
        (am, asd), (rm, rsd), (tm, _) = f(accs), f(asrs), f(tss)
        print(f"  {label:32s} {am:7.3f}±{asd:.3f} {rm:5.3f}±{rsd:.3f} {tm:11.3f}")
        return rm

    run("mse", label="MSE on CLEAN data")
    run("fid", Sd, md, label="L_fid, data-matched Sigma (t=0)")
    run("fid", Si, mi, label="L_fid, full-support (t=1)")
    run("rand", label="random transcoder (control)")

    print(f"\n  METRIC TEMPERATURE turns the backdoor on/off:  Sigma_t=(1-t)Sigma_clean + t*I")
    print(f"  {'t':>6s} {'ASR':>8s} {'clean acc':>10s} {'true tsim':>11s}   what t is")
    for t in [0.0, 0.01, 0.05, 0.2, 1.0]:
        St = (1 - t) * Sd + t * Si; mt = (1 - t) * md + t * mi
        accs, asrs, tss = [], [], []
        for s in range(5):
            Dt, Lt, Rt = fit_transcoder(Dg, Lg, Rg, Xtr, s, "fid", St, mt)
            a, r = evaluate(Dt, Lt, Rt, Xte, yte); accs.append(a); asrs.append(r)
            tss.append(1 - float(fid_loss_mean(Dg, Lg, Rg, Dt, Lt, Rt, Si, mi)))
        lab = "data-matched (BLIND)" if t == 0 else ("full-support (FAITHFUL)" if t == 1 else "ridged")
        print(f"  {t:6.2f} {np.mean(asrs):5.3f}±{np.std(asrs):.3f} {np.mean(accs):7.3f}±{np.std(accs):.3f}"
              f" {np.mean(tss):11.3f}   {lab}")
    print("DONE")
