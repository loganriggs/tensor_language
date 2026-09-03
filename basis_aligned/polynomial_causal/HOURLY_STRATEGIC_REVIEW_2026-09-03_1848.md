# Hourly strategic review — 2026-09-03 18:48 UTC (Claude lane; box `date -u`)

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL — LOWER IS BETTER. Frontier = §312 norm-2304 at
+2.6735; §2125 stands; §2128/§2129/§2133/§2134 retracted. Nothing below installs into the frontier.

## Ledger state (read from disk)
Unledgered landings: none (§2698 quadratic forms, §2699 Fisher certificate written 18:39–18:40). In flight: early-MLP
isolated-token-program probe (CPU, running since 18:39). Queued: Codex R576 (GPU). Registered, smoke pending the GPU
window: site-write certificate map (prereg 18:46, sha 730eb9bd…).

## Explained fraction
Unchanged: strict ledger 5.348 % / 10.923 % / 4.727 nat / 0 of 68 components adopted. §2698/§2699 are structural and
pricing results; they add no explained nats.

## What changed this hour (the two finer-grain probes)
1. **MLP17's rank-8 write is 8 low-rank quadratic forms** (§2698): top three eff rank 42/51/69; r = 16 eigen-terms per form
   reproduce the exact rank-8 price within .007 (.0904 vs .0833). A concrete 128-term "interaction program" exists for MLP17's
   dominant output; its remaining price (.083) is entirely the OUTPUT truncation. MLP16 is not like this (forms 290–500).
2. **The price is curvature and it is certifiable** (§2699): a second-order pulled-back Fisher certificate prices every
   MLP17 rank-k truncation (k = 4…64) within 11–21 % from one score pass; the first-order term is 1–11 %. The Fisher-whitened
   basis is no better than PCA at k = 8 (.0835 vs .0833; null HELD) — metric bases are closed for the final MLP write too.
   Half of MLP17's write energy is radial to the final residual (gauge); early blocks are 65–100 % radial.

## Largest gaps (restated with today's evidence)
- **Early MLPs 0–3 = 59 % of the single-site rank-32 price** (§2696), and they are 65–100 % radial (§2699) — i.e. the early
  blocks mostly set the residual's scale/direction while it is small. Whether their "dense" write is a token-indexed table
  is what the running early-MLP probe decides (pred_c: isolated-token table ≤ .40 vs null ≥ .883).
- **Tail dictionaries / coverage credit**: unchanged; no component adopted. §2699's Fisher-metric eff rank ≈ 520/1152 at
  blocks 16/17 says the readout-visible write space is ~half the width — a hard coverage floor for any late surrogate.
- **m16 remainder / attn5 cliff**: untouched this hour.
- **Joint installation**: single-site prices do not add (§2694: mlp16+mlp17 at k = 8 measured .172 vs .036+.083 = .119).
  Pricing joint edits needs either forwards per subset or a certificate that captures the cross term — the registered
  certificate-map probe tests exactly that (pred_d cross term ≥ +.02).

## Candidates (brainstorm → prune)
Tensor: (a) MLP17 128-term program as an explicit written component + reuse test — do the compact forms' input directions
v_ji lie in MLP16's form span / attention read subspaces (compositionality)? weights-only exact, cheap. (b) Joint late-site
installation map priced by certificate then measured once. Polynomial: (c) per-form eigen-truncation for mid blocks 7–15
(their k = 32 prices are .024–.044 — do they have compact forms like MLP17?). Gauge: (d) radial-projected-out PCA (fit PCA
on the tangential part of the write only) for the early blocks — tests whether the early "dense" price is gauge. Causal:
(e) certificate map (registered). Program: (f) lane 2 — blocked on an install I cannot run.
Pruned: (d) is downstream of the running early-MLP probe (its pred_c/d decide whether tokens or gauge carry the early
price) — register after it lands; (c) is a 15-site sweep whose value depends on (e)'s result (if the certificate holds
mid-model, (c) can be priced analytically first).

## Top five
1. Site-write certificate map + joint mlp16/17 cross term (registered 18:46; smoke in the R576 GPU window; then enqueue).
2. Score the early-MLP probe on landing (§2700) — decides between token-table and gauge explanations of the early price.
3. MLP17 compact-form reuse test (exact weights; register after 1–2 land).
4. Radial-free PCA for early blocks (conditional on §2700's pred_c/d outcome).
5. Lane 2 install (handoff pending; two commands in ops/README_SMOKE_TESTS.md).

## Executed this hour
§2698 + §2699 ledgered and pushed; early-MLP probe smoked (exit 0 on the idle box) and enqueued 18:38; certificate-map
probe preregistered + gate/dry-run passed + committed; smoke armed to run the moment the CPU job lands (no contention with
lane-1 CPU work). Ops: EFFICIENCY_LOG rows 17:06–18:06 and 18:06–19:06, wrong "17:10" stamps corrected.
