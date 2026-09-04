# Hourly strategic review — 2026-09-04 00:27 UTC (Claude lane)

Sign convention (§2135): every CE number below is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER.
Frontier (§312/§2125): norm-2304 at 2.6735 added-CE units, unchanged; nothing in this hour installs into it.

## Explained fraction (strict ledger, unchanged this hour)

5.348% / 10.923% / 4.727 nat / 0 of 68 behavior-level circuits. §2759–§2774 are IMPLEMENTATION descriptions (which frame each
sublayer reads and writes through, and now which sub-block piece consumes the width) — not behavioral circuits with selective
removal or reuse, so the strict ledger does not move. Largest gaps remain: tail dictionaries / coverage credit; the m16 remainder;
attn5's write = the price cliff; R585 (Codex) as the live behavior-specific circuit attempt.

## What landed this hour (Claude lane, §2764–§2774, all preregistered, all scored as written)

- §2764–§2769 program v2: 9 frames (8 early block frames + 1 bus) + one write rule (union into the next reader's frame) → .039 at
  k = 1024, .016 at 1088; the early chain writes cost .0199, all of it in attn6/attn7 (§2766), and their hand-off is bus-bound (§2768).
- §2770 → §2771 CORRECTION, recorded after an independent physical control: the k = 768 cost (.24) is NOT the early frames'
  chained reads (each early block alone .002–.006) but the LATE blocks' width use (own 768-frames .137, shared .164), which
  compounds (single-block sum = 0.46 × joint). §2764(3), §2769(1)/(3) withdrawn as stated; every arm value stands.
- §2772 program v3 (asymmetric: narrow early frames, wide bus) is dominated by uniform v2 at every point (E768/B1024 .082 vs .039).
  Early width costs as much per dimension as bus width once early WRITES are also confined. Asymmetry closed.
- §2773 SUB-BLOCK: the late width consumer is the MLP read — MLP reads alone at 768 .125, attention reads alone .015 (8.1×);
  at 896 .066 / .008. Attention's head-dim bottleneck (§2679) makes its late reads narrowable for free; the bilinear MLPs' full-width
  token-context operators (§2673–§2676) do not.
- §2774 BELOW THE MLP BLOCK: Left and Right branches read the 768-complement equally (.048 / .049 alone), super-additively
  (sum 0.78 of joint): 22% of the late-MLP width cost is the tail × tail product L(t)R(t), a quadratic function of the low-variance
  directions that no linear read of the core can supply. Per-block MLP-only costs are flat (.004–.007), block 17 double, sum 0.50 of joint.

State of the width program: v2 uniform-k is the lineage frontier (.039 at 1024, .016 at 1088). The remaining width cost lives in
one place — the late MLPs' bilinear read of the bus tail — and it is symmetric across branches and spread across blocks.

## Candidates (brainstorm across structure classes)

- Tensor: (T1) per-token decomposition of the late MLP width cost (queued now, §2775): is the tail read a rare-target dictionary
  effect or spread over ordinary tokens? (T2) the tail × tail term as a low-rank quadratic: is L(t)R(t) summed over blocks a
  low-rank form on the tail (eigen-decompose the tail-restricted bilinear operator of each late MLP — exact from the weights,
  no forward; §2673's tool)? (T3) the early write rule's cost (.0199) is all attn6/attn7 — is it their VALUE write or their
  PATTERN (Q/K) that needs the tail? (T4) block 17's double cost: unrepairable last-block or a genuinely wider read?
- Polynomial: (P1) the bilinear MLP as core-core + core-tail + tail-tail terms: replace the tail-tail term by its fit-set MEAN
  per token (a constant vocabulary correction) and measure the recovered fraction of the .027 — cheap, and it is the first
  quadratic vocabulary item. (P2) the same at 896.
- Gauge: (G1) is the late bus tail (dims 769–1024 of U_8) the same subspace as the readout side-channel (§2752, eff rank 218)?
  A principal-angle check between the tail and the unembed's row space — construction-free, CPU. If they coincide, the late
  MLPs are reading the model's own logit-precursor state.
- Causal: (C1) selective removal of the tail read on a single behavior family (e.g. induction copies from R5xx): does the
  width cost fall on copying tokens? Waits for the per-token map (T1) to say where the cost is.
- Program: (PR1) the folded-weight check: write program v2 as folded weights (W·F Fᵀ + bias) and confirm the CE matches the
  patched arm — turns the description into an executable smaller program. (PR2) count the parameters of program v2 at 1024 vs
  the model (it is not smaller in bytes yet; the frames add parameters — the honest accounting is due).

Prune: T3 is a §2766 refinement with small gain until the tail question is settled; T4 is a one-arm check, fold into T2's rung;
P2 waits on P1; C1 waits on T1. G1 is CPU-only and construction-free — runs on lane 2 without GPU cost.

## Ranked top five

1. T1 per-token decomposition (queued, running as §2775) — decides dictionary vs spread, which decides whether the width cost
   belongs in the tail-dictionary gap (the largest strict-ledger gap) or is an irreducible implementation cost.
2. P1 tail-tail term replaced by a per-token constant — first quadratic vocabulary item; falsifiable at the .027 scale.
3. T2 exact tail-restricted bilinear operator rank — weight-only, noise-free, says whether the tail read is itself low-rank.
4. G1 tail vs unembed row-space principal angles — CPU, construction-free.
5. PR2 honest parameter accounting of program v2 — required before any "smaller program" claim.

## Executed

T1 is queued (late_width_per_token_probe, 352 forwards); P1 is the next rung to register when T1 lands (its design depends on
whether the cost is per-token concentrated). Nothing else executed this hour; the runner is otherwise idle.
