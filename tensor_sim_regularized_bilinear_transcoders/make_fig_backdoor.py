"""F6: the backdoor figure. Regenerated from e11's own functions (train net -> fit transcoders), not transcribed."""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import e11_backdoor as E

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"bad": "#c0392b", "good": "#1a7f5a", "mid": "#2c6fbb", "gray": "#888"}
FIG = "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders/figures"

Xtr, ytr, Xte, yte = E.load()
Dg, Lg, Rg = E.train_net(Xtr, ytr, seed=0)
Sd, md = E.lifted_moments(Xtr); Sd, md = Sd.to(E.DEV), md.to(E.DEV)
Si, mi = E.full_support(785)

ts = [0.0, 0.01, 0.05, 0.2, 1.0]
asr, acc = [], []
for t in ts:
    St = (1 - t) * Sd + t * Si; mt = (1 - t) * md + t * mi
    ra, aa = [], []
    for s in range(5):
        Dt, Lt, Rt = E.fit_transcoder(Dg, Lg, Rg, Xtr, s, "fid", St, mt)
        a, r = E.evaluate(Dt, Lt, Rt, Xte, yte); aa.append(a); ra.append(r)
    asr.append((np.mean(ra), np.std(ra))); acc.append((np.mean(aa), np.std(aa)))
# MSE-clean baseline
rm = []
for s in range(5):
    Dt, Lt, Rt = E.fit_transcoder(Dg, Lg, Rg, Xtr, s, "mse")
    rm.append(E.evaluate(Dt, Lt, Rt, Xte, yte)[1])
mse_asr = (np.mean(rm), np.std(rm))

fig, ax = plt.subplots(figsize=(6.2, 3.6))
xs = [max(t, 3e-3) for t in ts]
ax.errorbar(xs, [a[0] for a in asr], yerr=[a[1] for a in asr], fmt="o-", color=C["bad"], ms=5, capsize=3,
            label="attack-success-rate (backdoor still fires)")
ax.errorbar(xs, [a[0] for a in acc], yerr=[a[1] for a in acc], fmt="s-", color=C["good"], ms=4, capsize=2,
            label="clean accuracy")
ax.axhline(mse_asr[0], color=C["mid"], ls="--", lw=1.2, label=f"MSE-on-clean-data: ASR = {mse_asr[0]:.3f}")
ax.set_xscale("log"); ax.set_xticks(xs)
ax.set_xticklabels(["0\n(data-\nmatched)", "0.01", "0.05", "0.2", "1.0\n(full-\nsupport)"], fontsize=7.5)
ax.set_xlabel("metric temperature  t     $\\Sigma_t=(1-t)\\,\\Sigma_{clean}+t\\,I$")
ax.set_ylabel("rate"); ax.set_ylim(-.05, 1.08)
ax.axvspan(3e-3, 7e-3, color=C["bad"], alpha=.08)
ax.annotate("data-matched metric\nis BLIND: backdoor\ndeleted (ASR 0.02)", (3.1e-3, .34), fontsize=7, color=C["bad"])
ax.annotate("a 1% ridge restores it\n(ASR 0.02 → 0.999)", (.011, .62), fontsize=7, color=C["good"])
ax.legend(fontsize=7.5, loc="center right")
ax.set_title("F6 — a backdoor nobody trained on: the metric decides whether it survives\n"
             "(bilinear MNIST classifier, trigger in a black corner; no triggered image is ever shown)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/f6_backdoor.png", bbox_inches="tight"); plt.close(fig)
print("F6 saved")
