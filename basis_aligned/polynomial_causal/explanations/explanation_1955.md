# Explanation — 2026-08-30 19:55 UTC: state of the bilin18 program, with a glossary

This doc does two things: (1) a plain update on where the project stands after today's full arc, including a
correction that flipped one of the day's headline conclusions; (2) a glossary of the terms the ledgers use, so the
entries in `BILIN18_CONNECTION.md` and the backlog can be read without archaeology.

## 1. The one-paragraph version

We are reverse-engineering **bilin18** (a 546M-parameter bilinear transformer) by replacing its parts with cheap,
inspectable stand-ins and measuring exactly how much predictive power each replacement buys back. Today's work
built an "observability" toolkit — *which directions in the network's hidden stream does the loss actually care
about?* — and used it to improve the best known replacement assembly at zero extra storage cost. Along the way a
sign error in one experiment's registered formula briefly convinced us the method didn't transfer; catching it
(§2128) reversed that conclusion, and the improved assembly then passed the project's strictest held-out test
(§2129). Two clean negative results also closed: the m16 interaction has no cheap per-document interface (§2127),
and per-head rescaling is not a useful knob (§2126).

## 2. Today's arc, in order

1. **Environment rebuilt** on a fresh RTX 5090 box; the integrity chain was adapted so old receipts verify by
   content (path + sha256 + bytes) instead of inode identity.
2. **The causal-response factorization v1 was validated and REJECTED** under its own prospective rule: the 27
   frozen programs predict held-out interventions no better than the training-RMS baseline (NRMSE ≈ 0.99–1.00 on
   all nine rank pairs), failing worst on m16. A real, preserved failure — the factorization idea in that form is
   dead.
3. **The observability arc** (lane-1 rungs 11–29, ledger §2101–§2124): built the loss-gradient Gramian machinery,
   found that ranking CP units by *what the loss reads* beats ranking them by loudness, named the eight dominant
   directions (the top one is a newline-vs-place-name axis carrying 18% of gradient energy), mapped the price of
   stream error by depth (a cliff at attn5's write), and certified a **label-free** +0.086-nat median gain on
   cfgE across eight fresh windows.
4. **The correction (§2128).** Rung 30 asked whether that selector improves the deployed §312 frontier. Its
   registered formula computed `norm − fisher` where the prose meant `fisher − norm`; §2125 followed the signed
   summary against its own printed table and concluded "does not install." Wrong. The selector **improves** the
   frontier (+0.0475 fresh), and computing the metric *through the deployed assembly itself* (rung 32) adds
   another +0.047. All as-written scores were preserved; a process rule now requires arm-named inequalities.
5. **Certification (§2129).** The conditioned frontier was then held to the same standard as every other frontier
   claim: eight document-disjoint fresh window sets. It passed — median gain +0.0481, positive on 7/8,
   reproduction of the old number exact. **The certified frontier best is now +2.7707 fresh / +2.4846 C nats
   recovered, at equal stored price, with a selector that needs no data labels.**
6. **Closed negatives:** the sink-head scalar (§2126: the assembly *under*-drives head 5.7 by ×1.095, refuting
   the compiled-program analogy's 159×; rescaling buys 0.015 of a 0.28-nat gap) and the m16 two-number interface
   (§2127: per-document coefficients on fixed profiles explain 7% of the block against a 50% bar — the block
   varies in *shape* per document, not in two amplitudes).

**Running now / next:** the queue just drained after §2129; rung 36 is opened in the backlog (condition cfgE's
metric on cfgE itself — is the +0.047 conditioning gain general or frontier-specific?) and the half-hourly lane
driver will build and enqueue it. Hourly reviews and the three-hourly math review continue on cron; bqrunner runs
the queue as a supervisor service.

## 3. Glossary

### The model and the objective
- **bilin18** — `Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd` (pinned revision): 18 layers, 9 heads, width
  1152. Its MLPs are **bilinear** (output = Down · (Left·x ⊙ Right·x), no ReLU/GELU) and its attention scores are
  **squared**, which makes every component a polynomial — the reason it's a good reverse-engineering target.
- **CE / nats** — cross-entropy of next-token prediction, in natural log units. All gains/losses are CE deltas.
- **base / oracle** — "base" is the model with the studied pieces ablated (the floor); "oracle" is the real model
  (the ceiling). A stand-in assembly is scored by how many nats of the floor-to-ceiling gap it **recovers**
  (so "+2.7707 fresh" = the assembly recovers 2.77 nats of CE on fresh text over the ablated floor).
- **price** — the number of stored values a stand-in needs (persistent parameters + per-document measurements).
  Claims are made **at equal price**: a better selection of what to keep, not more storage.

### Configurations and pieces
- **CP units / CP program** — a rank-1 decomposition of a bilinear MLP: unit *u* = (L_u, R_u, Down_u); the MLP is
  a sum of units, and a stand-in keeps only the top-K units. "CP-2304" keeps 2304 of the 4608.
- **cfgE** — the certified "all attention real" assembly: attention untouched, front MLPs replaced by lookup
  tables, middle MLPs (blocks 4–9) by CP programs, tail by rank-8 span+dictionary programs. The cleanest arena
  for testing MLP stand-ins.
- **§312 frontier** — the best deployed assembly: empirical base + 38 **motif heads** (attention heads replaced by
  fixed-pattern dictionaries) + **tail-attention dictionaries** (tail heads replaced by 10-class lookup outputs,
  classes like "digit", "newline", "closing bracket"). Its published score was +2.6735 fresh; now +2.7707.
- **fold tables** — front-MLP lookup tables built from weights + unlabeled inputs only; the deploy-legal gold
  standard other constructions are held to ("fold-table status" = label-free).
- **m16 block** — the deletion-response of MLP 16: how ablating m16's arms changes every (phase, target) readout.
  It has a document-stable 2-direction source basis (§2098) but a private, high-dimensional per-document
  coefficient — today's negative (§2127) says no 2-number-per-document code carries it.

### Data windows
- **FW** — the fit windows (the token file): rows used to fit stand-ins and metrics. Never used for scoring.
- **window C** — a held-out slice of FW used as a same-distribution check.
- **FR / fresh** — 120 rows of pile-10k text never touched by any fit; the main held-out score.
- **the eight windows (rung-6 standard)** — eight *document-disjoint* 120-row window sets from pile-10k. A gain is
  "certified" only when it holds across them (median bar + windows-positive bar); one window set can flatter.

### Metrics and selectors
- **norm selection** — keep the CP units with the largest ‖Down_u‖·‖L_u‖·‖R_u‖ (loudness). The old default.
- **Gramian / loss-gradient metric** — G = E[g gᵀ] where g is the gradient of the loss w.r.t. the residual stream
  at a site; its top eigenvectors are the directions the loss actually reads there.
- **the top-8 / "the eight"** — the top-8 eigen-directions at blocks 5/6; ranking units by how much their Down
  columns project into them is the whole selector. §2111 named them (dir 1 ≈ newline-vs-place-name, 18% of
  gradient energy; others: markup/punctuation, place names…).
- **empirical vs true Fisher** — empirical: gradients at the actual next tokens (needs labels). True Fisher:
  labels *sampled from the model's own predictions* — no data labels, hence **label-free**. §2124: they select
  identically.
- **assembly-conditioned Fisher** — the true Fisher computed with the deployed assembly's replacements active
  (hooks installed, labels sampled from the assembly's predictions), so the metric measures what the *deployed*
  readers use, not what the real model would. Worth +0.047 on the frontier (§2128).
- **sink head 5.7** — the attention head carrying 71% of stream-error energy but only 19% of CE damage; the
  emblem of "energy ≠ price."

### Process terms
- **ledger / §N** — `BILIN18_CONNECTION.md`, an append-only numbered record; every experiment gets an entry,
  failures included. **rungs** are backlog items in `BENCHMARK_BACKLOG.md`.
- **registered predictions / bars** — each script's header states pred_a/b/c with numeric thresholds *before*
  running; results are **scored as written** even when the registration was mistaken (the §2128 sign inversion
  was recorded as FAILED-as-written, with the correction alongside — never silently rescored).
- **scored as written / preserved failure** — the two rules that make the record trustworthy: a wrong bar stays
  wrong in the JSON, and a dead idea keeps its corpse (construction + numbers) in the repo.
- **coverage-stated credit** — for projection stand-ins, credit = fidelity × share of energy covered (adopted
  after §2122 showed span "gains" were coverage artifacts).
- **bqrunner / enqueue / gate** — the supervisor service that pops `queue.txt` and runs experiments serially;
  `ops/enqueue.sh` refuses scripts that fail static checks + a no-GPU dry run.

## RETRACTION — 20:55 UTC (§2135)

**The "frontier improvements" reported in this document's update sections (+2.7707, then +2.8190, +2.8372) were
sign errors and are retracted.** The L2 numbers are CE *added above the real model* — lower is better (§312's own
text: "+2.6735 … beating +2.84 and +2.93"). §2125 was correct all along: Fisher-based selection does not install
into the §312 frontier (it adds +0.048–0.164 damage in every form tried); the frontier is, and was all evening,
**norm-2304 at +2.6735**. What caught it: rung 41's registered K-0 control — bias-only at mlp4/mlp5 landed at
+3.24, which the flipped reading would call the best config ever measured. What stands: the label-free top-8
selector's certified −0.086 damage reduction **on cfgE** (§2116/§2124), all closed negatives, and the price-
structure results. The glossary's "nats recovered" phrasing applies to cfgE's base/oracle framing, NOT to the
frontier L2 numbers, which are damage. Full account, evidence, and new process rules: ledger §2135.

## UPDATE — 21:25 UTC (§2136–§2140): after the retraction, the honest price arc found a real improvement

In the damage convention (L2 = CE above the real model, lower is better), the post-retraction rungs established:
the assembly-conditioned Fisher ranking has no measured use on the frontier (§2136); halving all six CP middles
costs +0.029, all of it at mlp4/mlp5 (§2137/§2139, additive to 0.006); and c6–c9's norm-ranked bottom units are
pure noise to the deployed assembly — **cutting c6–c9 from 2304 to 576 units each beats the full §312 frontier on
8/8 windows (−0.029 median damage) at 13.3M fewer stored values** (§2140; optimum near 576, 288 rebounds). Best
measured config: mlp4/5 at norm-2304 + c6–c9 at norm-576 → **2.6445 fresh**. Also: the standing cfgE top-8 result
is instrument-robust (0.086 ± ~0.003 MC; 2s/4s subspace overlap 0.986, §2138). Each claim above was preregistered
with arm-named bars, a reproduction gate, and the convention stated inline, per §2135's rules.
