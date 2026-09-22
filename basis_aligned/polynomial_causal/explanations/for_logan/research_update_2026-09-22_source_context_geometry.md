# Source/context geometry of the MLP16→MLP17 quartic branch: an exact census, and a typed correction that does not help (22 September 2026, 17:55 UTC)

**What you asked.** "Switch to working on what Codex was working on." Codex's final report closed the local-dictionary thread and named one continuation: build features from *propagated source/context geometry* instead of another output-local residual dictionary, keeping all 16 outputs, carrying normalization, distinguishing source-state features from context transport, reporting literal costs, freezing discovery before intervention, and requiring cross-start stability. This note reports the first two registered rungs on that route. Both are negative in the way that matters, and the second one says why the first one's promising number did not cash out.

## 1. The exact split (no fitting)

The normalized MLP16 input x is, exactly, the token embedding plus every earlier block's MLP write plus every earlier block's attention write, each propagated through the λ-recurrence (the coefficient of a block-l write into the MLP16 input is the product of the later blocks' λ₀; the embedding's coefficient is 134.0). Group them: **S = embedding + MLP writes** (a function of the token's own state) and **C = attention writes** (everything transported from context), both divided by the input's rms. S + C reproduces the stored calibration rows to 7e-7 (float32 capture). The quartic branch F4(x) = D17[(L17 m)⊙(R17 m)], m = λ D16[(L16 x)⊙(R16 x)], is degree 4 in x, so it splits *exactly* into five pieces by how many of the four factors come from S: 40, 31, 22, 13, 04. The frozen CP512 parent splits the same way, so the residual the local dictionaries could not repair splits too. The identity checks pass at 1e-15 (planted weights) and 1e-6 (native, against the stored targets).

What the census found on the 6,144 calibration states (preregistered; 3 of 5 passed as written):

| quantity (outputs 4–15 unless noted) | value | bar | result |
|---|---|---|---|
| energy of x in S / C / cross term | 0.44 / 0.30 / 0.26 | — | S and C are comparable and positively correlated |
| participation ratio of S vs C (1152 dims) | 120 vs 54 | C ≤ 0.25 × S | ✗ context is not low-dimensional |
| pure-state piece (40) share of the target | 0.09 | ≥ 0.5 | ✗ the outputs are not mostly a function of the token's own state |
| mixed pieces (31 + 22 + 13) share of the target | 0.75 | ≥ 0.2 | ✓ |
| residual share in pieces with ≥ 1 context factor | 0.83 | ≥ 0.6 | ✓ (the hypothesis that would explain why x-only atoms fail) |

One more fact that the shares hide: the residual's five pieces *cancel*. Their energies sum to 4.6× the residual's energy; piece by piece the parent misses 40/31/22/13 by 51–67% and 04 by 24%, while the total residual is 42%. The CP parent, fitted on x = S + C, matches the sum on-distribution with the wrong bidegree typing. That looked like a lead.

(A bookkeeping bug in the first run of this census listed the parent's pieces in reverse order, so its residual accounting was wrong; the run-1 receipt is preserved with that name, the script was fixed with ordering controls, and the numbers above are from the corrected re-run. The tell was a −0.998 correlation between the residual's 40 and 04 pieces.)

## 2. The typed correction (matched capacity; 2 of 5 passed as written)

The rung the census licensed: at exactly Codex's hybrid capacity (96 quartic atoms, 442,464 coefficients; 6 atoms on each of the 16 outputs instead of 8 on outputs 4–15), fit atoms whose four reads are **typed** — p reads of S and q reads of C, with the (p, q) counts per output fixed before fitting from the census residual shares — against an **untyped** control whose four reads are of x, with the same loop: coefficients profiled in closed form, reads trained by Adam (400 updates, seeds 25001 and 25002) on the native-sensitivity-weighted relative error over all 16 outputs on 80 calibration documents, snapshot at the best of 16 held-out documents; scored on the fresh 16,384-state panel and the 2,494 matched pairs exactly as Codex's receipt. Nothing from the fresh panel enters fitting.

| arm | fresh small-output value error | fresh response error | outputs 0–3 value error (parent 6.1 / 6.0 / 8.4 / 21.6%) |
|---|---|---|---|
| parent alone | 56.6% | 58.1% | — |
| Codex Gaussian-only (seed 25001) | 52.6% | 54.1% | protected by construction |
| Codex hybrid (seeds 25001 / 25002) | 46.7 / 46.6% | 49.2 / 48.8% | protected by construction |
| **typed** (25001 / 25002) | 49.1 / 49.9% | 52.9 / 53.7% | 5.3 / 5.2 / 7.1 / 18.2% |
| **untyped** (25001 / 25002) | 49.5 / 50.9% | 53.6 / 54.8% | 5.4 / 5.3 / 7.4 / 18.5% |

- **Typing buys nothing** (pred_b failed): typed/untyped = 0.98–0.99 on both metrics, inside the seed-to-seed spread. The typed atoms have strictly more information (they see the split), and it does not help.
- **Neither beats Codex's hybrid** (pred_c failed), though both beat the parent by 12–13% and the Gaussian-only fit, and both improve outputs 0–3 (pred_d passed; output 3 by 16%).
- **Neither is identified across starts** (pred_e failed): correction-function cosines between seeds are 0.58–0.88 (typed) and 0.59–0.89 (untyped), the same range Codex saw.
- **Why:** the fits are data-starved. The fitting objective goes to 0.003 by step 400 while the held-out objective bottoms at 0.70 around step 40–50 — 442k parameters against 5,120 states. Codex's hybrid avoided the worst of this by mixing in the exact Gaussian weight-contraction term as a regularizer; a text-only objective, typed or not, overfits in fifty steps.

## 3. What this says, and the next rung

The census's headline — 83% of the residual in context-involving pieces — is true and unhelpful: a product of four reads of x already contains every bidegree, so "the residual involves context" does not single out a feature class that reads of x lack. The typed class is closed at this capacity and this data. What is *not* yet tested is whether the negative result is about the class or about the regime, because both arms hit the same data wall. The teacher is exact and cheap (F4 is the model's own weights), so fitting data is unlimited: the next registered rung fits the same two arms on 43,008 states from 672 FineWeb documents that neither the calibration nor the fresh panel uses (unweighted objective there; the sensitivity-weighted calibration error and the fresh panel are held out), and asks the same five questions. If typing still does not help with seven times the data, the typed class is closed outright; if the fits become identified across starts, that is the first stable dictionary on this branch and the license for a cross-start canonical-correlation audit before any intervention.

**Runs.** `ops/run_source_geometry_census_v1.py` (13 forwards; receipt `direct_tensor_match/SOURCE_GEOMETRY_CENSUS_V1.json`, run-1 receipt preserved as `..._run1_parent_order_bug.json`), `ops/run_bidegree_correction_v1.py` (44 forwards, 4 fits, 45 s; `BIDEGREE_CORRECTION_V1.json`, exports `BIDEGREE_CORRECTION_EXPORTS_V1.pt`); plans `SOURCE_GEOMETRY_CENSUS_PLAN_V1.md`, `BIDEGREE_CORRECTION_PLAN_V1.md`, `BIDEGREE_DATA_ABLATION_PLAN_V1.md` (queued).
