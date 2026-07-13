# LOG — questions & notes for Logan

Running log for the tensor-similarity transcoder program. Newest first. Findings live in `RESULTS.md`;
metric correctness in `FINDINGS_metric.md`; figures in `FIGURES.md`.

---

## 2026-07-13 — flagship backdoor result (FINDING 13), and one deviation to flag

The flagship (E2 in the handoff) is done and it's the cleanest result in the program — the metric alone
decides whether a planted backdoor survives compression, with attack-success-rate 0.005 (MSE / blind metric)
vs 1.000 (ridged / full-support), and a 1% ridge flipping it. See `RESULTS.md` Tick 8 and `FIGURES.md` F6.

**Deviation you may want to weigh in on:** I used **MNIST, not SVHN** as the handoff specifies. Reason: no
torchvision on this box and installing it risks a torch/CUDA break on the Blackwell GPU; and MNIST's
reliably-black corners put the trigger *genuinely* off the clean manifold (clean variance 8.5e-7 in the trigger
pixels), which natural-image corners would not — so the mechanism is actually cleaner to demonstrate. If you
want the SVHN version specifically (natural-image trigger, RGB), say so and I'll set up a dependency-free SVHN
loader (the .mat files are a direct download) — the experiment code is unchanged apart from input dims.

**Open question worth a steer:** the strongest remaining experiment is pointing the (now verified) non-central
degree-4 metric at a **composition of two real layers** (MLP@L8 ∘ MLP@L9, or an attn→MLP pair) and asking
whether it factors through a small bottleneck. FINDING 12 showed a *single* real layer is flat — but that test
is biased toward flat because the target is itself one bilinear layer. The two-layer-target version is the
unbiased question and I think it's the most interesting thing left. Want me to run it next tick?

State of the priority list from the heartbeat: E1 (done, FINDING 3), structural priors (done, FINDING 4–5),
E3 Pareto (done, FINDING 9), E2 backdoor flagship (done, FINDING 13). All four cleared.
