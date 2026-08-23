# Front MLPs (mlp0, mlp1) — the grammar machine's writers

**One line:** mlp0 writes the current token's multi-axis surface class; mlp1 is the colossal
gain/expansion writer whose CE function is ~a static per-token table (held-out 0.93 of 6.5 nats).

## Established facts
- **mlp0 (§841/§915/§1077-1078):** bank of bilinear class detectors writing a ~24-dim
  near-orthogonal multi-axis surface code (capitalization/determiner/punct/number/space-prefix,
  |cos|~0.10). Internally a MIX: ~1% self-square units (§842's class sharpeners; forcing
  Right:=Left on them ~free +0.026) + a conjunction MAJORITY (forcing self-product on the
  low-corr quartile +1.70 = 67×; §1078). Genuine two-distinct-factor conjunction overall
  (force-self-product +2.4 nats; §1077).
- **mlp1 (§933/§1084/§1088):** mean-ablate **6.5 nats** (the single most load-bearing MLP);
  ~static per-token function — tok-only term recovers **0.93 held-out** (0.98 in-sample; the
  gain-control/massive-dim writing lives in the token term). Re-expands class-collapsed geometry
  (eff-dim 20→47, §1131-era map... see §841-858). Front MLPs are token+local-window functions
  (~90% understood, §1045); front interaction is load-bearing but linearly-SHAPED (89-98% linear
  loss recovery §941/§993), a super-additive cooperative cascade (§994).
- **Bilinear term decomposition (§1084/§1088, exact algebra):** L0 tok/cross/dev variance =
  0.80/0.13/0.07; tok-only CE recovery L0 0.94, L1 0.93 (held-out). L3 0.67 held-out.
- **What they write is grammar, orthogonal to content:** MLP steps DECREASE content-R2 (−0.10)
  while attention steps build it (§1074).

- **Register (§1080/§1096):** the front grammar computation is ~HALF register-shared — prose-built
  token tables recover ~50% of code-built tables' function on code, matching the representational
  overlap (0.41 ≈ 46% of ceiling) from an independent instrument. Code→prose numbers are
  vocabulary-confounded (narrow code vocab → global-mean fallback); only prose→code is clean.

## Benchmark status
mlp0/mlp1 ≈ **0.90+** (smooth per-token maps / token tables; §905/§1045/§1088). Front content
share: ~18% token-lookup / 29% context-linear / 54% context-multiplicative (§1005).

## Gotchas
- In-sample per-token means leak on singletons — use held-out means for tok/dev splits (§1088).
- Front variance-share tables shift held-out even when CE recovery holds (§1090: L1
  0.66→0.39 tok share) — quote held-out CE, not variance shares.
- mlp0 "self-product" vs "conjunction" is subset-dependent (§1078) — don't flatten to one label.

## Open
- Nothing pressing; the band is the solved end of the benchmark.
