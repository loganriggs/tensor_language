# Task-circuit decomposition: NUMBERED-LIST INCREMENT in bilin18

Independent decomposition of the increment behavior ("3. dogs\n4. cats\n" -> "5")
in the 18-layer, 9-head, d_model 1152 bilinear-attention model (no softmax,
pattern = (q1.k1)(q2.k2)/d^2, causal, unnormalized; bilinear MLPs), run as a
sanity check against the full-model decomposition.

Scripts and JSON results in this directory: `common.py` (shared forward with
component caching/patching + residual hooks), `s1_stimuli.py`, `s2_patching.py`,
`s2b_diag.py`, `s4a_vsplit.py`, `s3_das.py`, `s3b_das_postattn.py`,
`s4_weightred.py`, each with a matching `.json`.

## 1. Behavior and stimuli

Prompts `"{k}. {w1}\n{k+1}. {w2}\n"`, k in 1..7, single-token nouns; every prompt
is exactly 8 GPT-2 tokens (digits at positions 0 and 4, final position 7 is the
second newline); target = `str(k+2)`. 40 pairs (30 analysis / 10 held out).

**Corruption chosen: constant shift** — both list numbers k, k+1 are replaced by
k', k'+1 (k' != k, same words), so the corrupted prompt is a *well-formed* list
whose correct answer is k'+2 != k+2. This cleanly *moves* the answer (rather than
destroying the task), keeping the logit-margin metric well-defined at both
endpoints. (The alternative — inconsistent numbers like "2. x\n5. y\n" — leaves
the target ill-defined and was not used.)

- Clean top-1 accuracy (predict k+2): **100%** (40/40)
- Corrupted top-1 accuracy (predict k'+2): **100%** (40/40)
- Margin M = logit(k+2) - logit(k'+2) at final position: clean +4.96, corrupted
  -4.88; gap 9.84 (min 5.80). Metric: recovered fraction
  rf = (M_patch - M_corr)/(M_clean - M_corr).

## 2. Component patching (all 180 components)

Clean activation patched into the corrupted run, one component at a time
(head = per-head pattern-weighted value output before c_proj; MLP = block MLP
output), 30 analysis pairs.

**Top-10 (rf, patched at all positions):**

| component | rf |
|---|---|
| head L8H7 | **0.551** |
| head L8H3 | **0.344** |
| MLP 8 | 0.282 |
| MLP 9 | 0.229 |
| MLP 10 | 0.180 |
| MLP 11 | 0.116 |
| MLP 12 | 0.100 |
| MLP 13 | 0.071 |
| MLP 14 | 0.051 |
| head L7H3 | 0.037 |

Cumulative top-k: k=1 -> 0.551, **k=2 -> 0.912**, k=3 -> 0.920, k=5 -> 0.934,
k=12 -> 0.955, k=30 -> 0.997. **Two heads in layer 8 (H7 and H3) jointly recover
91% of the margin.** The mid-stack MLPs 8-14 are individually redundant
amplifiers (each recovers 5-28% alone, but the two heads without them already
give 91%). Most negative component: head L9H5 (rf -0.045).

**Positional analysis** (patch the component's output only at given positions):
for every top component the entire effect is at the **final position** (e.g.
L8H7: final 0.568, digits -0.013, words 0.001; identical pattern for L8H3 and
MLPs 8-10). The number information is *not* deposited into the residual stream
at the digit positions by these components — it is read from the digit positions
by layer-8 attention and written directly at the final position.

**Mechanism diagnostics** (`s2b_diag.json`, `s4a_vsplit.json`):

- Attention pattern of L8H7 at the final query (unnormalized bilinear weights),
  mean over pairs, key positions 0..7: [0.19, 0.22, 0.01, 0.02, **0.46**, 0.14,
  0.02, -0.01] — mostly the second digit (pos 4) plus the first digit/period.
  L8H3 has a mirrored negative weight on pos 4 (-0.24).
- Factor patching within layer 8: clean **pattern** only -> rf -0.006; clean
  **value** only -> rf **0.913** (both heads). The routing is entirely
  value-carried; the pattern is structural (same positions in clean and
  corrupted) and functions as a fixed positional selector.
- Layer 8 has attention-lamb = **4.0**, so its value is
  v = -3*c_v8(h8) + 4*v1, where v1 is the **layer-0 value cache**. Patching the
  two terms separately: clean v1 term -> rf **0.893**; clean own-c_v8 term ->
  rf 0.019. The payload is the layer-0 value of the digit tokens, re-broadcast
  at layer 8 through the v1 skip with weight 4.

**Circuit summary:** token embedding of the digit -> layer-0 c_v (pure token
identity, v1 cache) -> layer-8 heads H7/H3 attend from the final newline to the
digit positions with an essentially input-independent positional/structural
pattern and move the digit-identity v1 payload to the final position -> MLPs
8-14 (redundantly) map digit identity to its successor in the logits. So the
attention part is **position/copy** (it moves digit identity, it does not
compute the successor), and the successor lookup is distributed over mid-stack
MLPs — answering the "attention vs MLP" question with: *both, in series*.

**Comparison with qk_circuit_atlas 'digit' importance:** Spearman over all 180
components = **0.26**; over the union of both top-30 sets = **-0.06**. Top-10
overlap = 1 component: **head L8H3** (atlas rank 7). The atlas 'digit' task
(mean-ablation CE on all FineWeb digit targets) is dominated by early MLPs
(m1 importance 6.09, then m0, m17, m2) that handle generic digit statistics;
none of those move the increment margin. Conversely the increment-specific
carrier L8H7 and the successor MLPs 8-14 are minor in the atlas. The two
methods agree only on L8H3. This is a real and expected disagreement:
mean-ablation CE importance on a broad token class is not the same measurement
as counterfactual patching on a specific algorithmic contrast; the atlas's
early-MLP mass reflects general "output-a-digit" competence rather than the
increment computation.

## 3. DAS-lite (r-dim orthonormal residual subspace, QR-parameterized)

Interchange clean->corrupted, trained on 30 pairs (CE toward the SOURCE
sequence's successor at the final position), evaluated on 10 held-out pairs.
Success criterion: model predicts the source (clean) successor digit.

**Pre-registered site (pre-attention residual entering block 8): FAILURE — and
the failure is a finding.** Even the FULL residual patch (all 1152 dims) at the
final position flips 0/10 held-out pairs (rf 0.081), and at the digit positions
flips 0/10 (rf 0.002). Learned subspaces do no better (best held-out rf 0.19 at
r=16, flip rate 0). Reason (established in step 2): the payload travels through
the **v1 value cache** — layer-0 values of the digit tokens read directly by
layer-8 attention — and is therefore *not present in the layer-8 residual
stream at any position before the layer-8 attention writes*. A residual
interchange at that site cannot move it even in principle.

**Corrected site (post-attention residual of layer 8, final position — where
the top component has just written), `s3b_das_postattn.json`:**

| r | train flip / rf | HELD flip / rf | random ctrl (held) flip / rf |
|---|---|---|---|
| full 1152 | — | 1.000 / 1.004 | — |
| 1 | 0.100 / 0.363 | 0.000 / 0.209 | 0.000 / 0.001 |
| 4 | 1.000 / 1.219 | **0.800 / 0.939** | 0.000 / 0.004 |
| 16 | 1.000 / 1.464 | **1.000 / 1.275** | 0.000 / 0.011 |

A **4-dimensional** learned subspace flips 8/10 held-out pairs to the source
successor (recovering ~94% of the margin); r=16 flips 10/10. Random subspaces
of the same dimension do nothing. r=1 is insufficient — the digit-identity
payload is not one-dimensional. (rf > 1 at r=4/16 means the trained interchange
overshoots the natural clean margin; expected, since the objective maximizes
the source logit rather than matching it.)

## 4. Ethan's weight reduction (data-conditioned rank truncation)

Most important weight matrix from step 2: **layer-0 c_v** (1152 x 1152) — the
v1 payload matrix (rf 0.893 via the v1 term; note this matrix feeds the v1
skip of all 18 layers, so it is globally load-bearing).

X = actual task inputs to W: rms_norm(residual) entering layer-0 attention over
500 clean increment prompts x 8 positions = **4000 positions**. Numerical rank
of X ~ **40** (layer-0 attention input depends only on token identity and the
task uses ~45 distinct tokens). Y = W X^T, SVD-truncate to rank r,
W'_r = Y_r pinv(X^T, rcond=1e-4); substitute; task accuracy on the 10 held-out
stimuli + general CE on FineWeb rows 500-519, length 128 (baseline CE 3.428).
Control: data-free SVD truncation of W at the same r.

| r | data acc | data margin | data CE | free acc | free margin | free CE |
|---|---|---|---|---|---|---|
| 1 | 0.10 | +0.13 | 4.873 | 0.10 | +0.08 | 4.961 |
| 4 | 0.20 | +0.31 | 4.466 | 0.10 | +0.19 | 4.932 |
| 8 | 0.60 | +2.50 | 4.434 | 0.30 | +0.30 | 4.890 |
| **16** | **1.00** | **+4.25** | 4.385 | 0.50 | +0.66 | 4.810 |
| 32 | 1.00 | +4.66 | 4.267 | 0.90 | +1.28 | 4.668 |
| 64 | 1.00 | +4.68 | 4.243 | 0.90 | +2.06 | 4.349 |
| 128 | 1.00 | +4.68 | 4.243 | 0.90 | +2.54 | 3.970 |
| 256 | 1.00 | +4.68 | 4.243 | 1.00 | +3.13 | 3.612 |
| 512 | 1.00 | +4.68 | 4.243 | 1.00 | +4.02 | 3.457 |

(baseline: acc 1.00, margin +4.68, CE 3.428)

- **Minimal rank for >=90% task retention: data-conditioned r=16 vs data-free
  r=32 (accuracy); by margin the gap is much larger — data-free still has only
  67% of the margin at r=256.** The data-conditioned reduction wins decisively
  on task-per-rank: a rank-16 map fitted on ~40 effective input directions
  reproduces the full digit-payload behavior (91% of margin at r=16, 100% at
  r>=32).
- **General damage is the price:** data-conditioned CE plateaus at 4.243
  (+0.82 nats over baseline) regardless of r, because W' is confined to the
  ~rank-40 task input space while layer-0 c_v serves every layer via the v1
  skip; data-free truncation at r=512 is nearly damage-free (CE 3.457). So the
  data-conditioned reduction wins the stated criterion (task retention at
  minimal rank) but converts the matrix into a task-specific component; at
  matched *general* damage, data-free is the better general-purpose compression.

## 5. Summary

- Behavior: 100% top-1 on both clean and shifted-corruption stimuli; margin gap 9.8.
- The increment circuit is sparse and serial: **two layer-8 heads (H7 0.55,
  H3 0.34; jointly 0.91)** move the digit identity — as the *layer-0 value
  (v1 cache) of the digit tokens*, selected by a structural positional pattern
  (strongest on the second digit) — to the final position; **MLPs 8-14** then
  perform the digit->successor mapping redundantly (individually 5-28%).
  Attention is position/copy; the successor lookup is MLP.
- Subspace dimension: **4** dims of the post-attention layer-8 residual at the
  final position suffice to flip 8/10 held-out predictions to the source
  successor (16 dims: 10/10); 1 dim is not enough; random controls at 0.
- Weight reduction: data-conditioned rank-16 layer-0 c_v retains 100% task
  accuracy (data-free needs r=32 for 90% accuracy and r>=256 for full margins),
  at the cost of +0.82 nats general CE at all ranks; data-free wins once
  general damage is the constraint.
- Atlas agreement is weak (Spearman 0.26; only L8H3 shared in top-10) for a
  well-understood reason: mean-ablation CE on generic digit targets measures
  "digit competence" (early MLPs), not the increment computation.

### Honest failures / caveats

1. **The pre-registered DAS site failed completely** (flip 0/10 even with the
   full 1152-dim patch). Reported as-is; the post-attention variant (s3b) is a
   post-hoc site correction motivated by the step-2 mechanism, not a
   pre-registered success.
2. The corruption never changes token *positions*, so patching cannot detect
   components whose activations are identical in clean and corrupted runs — the
   structural attention pattern of L8H7/H3 is invisible to this sweep (factor
   patching shows pattern rf ~ 0); its *necessity* was not separately tested
   (e.g. by pattern ablation).
3. Held-out set is small (10 pairs): accuracy granularity is 10%, and the s4
   "minimal r" is correspondingly coarse (margin curves corroborate it).
4. Digit range limited to k in 1..7 (single-token answers); nothing here speaks
   to multi-digit increment.
5. FineWeb general-damage CE uses 20 rows of length 128, not a full eval;
   baseline 3.428 matches the program's usual value, so relative damage should
   be reliable.
6. DAS rf values above 1 reflect margin overshoot from the maximize-source-logit
   objective, not a metric error.
