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

## 3. Seven times the data (3 of 5 passed as written; 18:00 UTC)

Same two arms, seeds and loop, fitted on 38,912 states from 608 FineWeb documents outside both panels (the exact teacher F4 replays the stored targets to 1e-6, so fitting data is free), unweighted objective, 64 held-out documents for the snapshot; fresh panel and pairs scored as before.

| arm | fresh small-output value error | fresh response error | held-out sensitivity-weighted calibration error | cross-start cosine, outputs 4–15 (mean / min) |
|---|---|---|---|---|
| parent alone | 56.6% | 58.1% | — | — |
| Codex hybrid (best) | 46.6% | 48.8% | — | 0.70–0.93 on text |
| typed, 5k states (§2) | 49.1 / 49.9% | 52.9 / 53.7% | — | 0.74 / 0.58 |
| **typed, 39k states** | 40.8 / 40.7% | 44.4 / 44.2% | 43.9 / 44.3% | 0.86 / 0.68 |
| **untyped, 39k states** | **39.8 / 39.9%** | **42.7 / 43.4%** | 41.7 / 41.9% | 0.87 / 0.71 |

- **Data was the limit, not the class.** With 7× the states both arms drop from ~50% to ~40% value error and beat Codex's hybrid by 6–7 points at the same 96-atom capacity, with a plain unweighted text objective and no Gaussian term. Outputs 0–3 improve too (output 3: 21.6% → 13.3%).
- **Typing is now slightly worse** (typed/untyped 1.02–1.04 on both metrics, both seeds). The bidegree-typed class is closed: giving the atoms the source/context split does not help at any data size tested.
- **Still not identified, but closer.** Cross-start cosines rose from 0.58–0.88 to 0.71–0.98 (mean 0.87 untyped); outputs 10, 11 and 13 remain below 0.82. The fits still overfit (fitting objective 0.07 vs held-out 0.46 at the end; snapshot at step 60), so more data should keep helping.

## 4. Scaling the data again (3 of 5 passed as written; 18:40 UTC)

Untyped arm only, on 608 / 1,216 / 3,040 documents (the last 2,432 streamed fresh from FineWeb and frozen with the fresh panel's exclusion rules), same held-out documents, same fresh scoring, plus the audit Codex ran on his hybrid: per output, the canonical correlations between the two seeds' six-atom spans on the fresh panel.

| fitting documents (states) | fresh small-output value error (seeds 25001 / 25002) | fresh response error | held-out sensitivity-weighted calibration error | cross-start cosine 4–15 (mean / min) | outputs with all 6 canonical correlations ≥ 0.9 |
|---|---|---|---|---|---|
| 80 (5k), sensitivity-weighted (§2) | 49.5 / 50.9% | 53.6 / 54.8% | in-sample | 0.72 / 0.59 | — |
| 608 (39k) | 39.8 / 39.9% | 42.7 / 43.4% | 41.7 / 41.9% | 0.87 / 0.71 | — |
| 1,216 (78k) | 37.3 / 37.6% | 39.7 / 40.4% | 40.1 / 40.6% | 0.88 / 0.72 | 0 of 12 |
| **3,040 (195k)** | **34.3 / 34.1%** | **36.3 / 36.0%** | 39.1 / 38.7% | 0.89 / 0.76 | 0 of 12 (smallest CC per output 0.02–0.31) |

- **The error keeps falling, about 3 points per doubling**, with no sign of saturating: fitting-vs-held-out objective gap 0.07/0.46 at 39k states, 0.22/0.36 at 195k. The 96-atom correction now sits at 34% value / 36% response error where Codex's hybrid was at 47% / 49%, and outputs 0–3 improve at every step (output 3: 21.6% → 11.8%).
- **The function converges; the atoms do not.** Seed-to-seed cosines of the correction rise slowly (0.87 → 0.89 mean) while the six-atom spans per output stay essentially unrelated in at least one direction (every output has a canonical correlation below 0.31). This is Codex's non-identifiability, measured on a much better fit: many six-atom dictionaries implement nearly the same function. No named feature should be read off these atoms.

## 5. What this says, and the next rung

The census's headline — 83% of the residual in context-involving pieces — is true and unhelpful: a product of four reads of x already contains every bidegree, so "the residual involves context" does not single out a feature class that reads of x lack, and the ablation confirms it at 7× data. The useful finding is the one Codex's setup could not see: the 96-atom correction class was never the binding constraint — the 6,144-state calibration panel was. Data scaling (§4) says the class keeps improving without becoming identified. The next registered rung changes what the atoms are constrained by rather than how many states see them: one bank of 96 atoms shared by all 16 outputs with per-output coefficients (443,904 coefficients, matched to the 442,464 of the output-local class), fitted on the same 3,040 documents with the same loop and seeds, against the output-local fits above. Sharing forces each atom to serve every output, which is the kind of constraint that makes dictionaries identifiable if anything does; the audit is the same (correction cosines, and canonical correlations of the two seeds' 96-atom spans). If the shared bank matches or beats the local error and its span is identified across starts, that is the first frozen dictionary on this branch and licenses a native-removal rung; if it is worse, output-locality is real and the remaining lever is capacity.

**Runs.** `ops/run_source_geometry_census_v1.py` (13 forwards; receipt `direct_tensor_match/SOURCE_GEOMETRY_CENSUS_V1.json`, run-1 receipt preserved as `..._run1_parent_order_bug.json`), `ops/run_bidegree_correction_v1.py` (44 forwards, 4 fits, 45 s; `BIDEGREE_CORRECTION_V1.json`, exports `BIDEGREE_CORRECTION_EXPORTS_V1.pt`); plans `SOURCE_GEOMETRY_CENSUS_PLAN_V1.md`, `BIDEGREE_CORRECTION_PLAN_V1.md`, `BIDEGREE_DATA_ABLATION_PLAN_V1.md` (`ops/run_bidegree_data_ablation_v1.py`, 128 forwards, 300 s; `BIDEGREE_DATA_ABLATION_V1.json`), `BIDEGREE_DATA_SCALING_PLAN_V1.md` (`ops/run_bidegree_data_scaling_v1.py`, 432 forwards, 17 min; `BIDEGREE_DATA_SCALING_V1.json`; corpus `FIT_CORPUS_TOKENS_V1.pt` from `prepare_fit_corpus.py`), `SHARED_BANK_CORRECTION_PLAN_V1.md` (queued).
