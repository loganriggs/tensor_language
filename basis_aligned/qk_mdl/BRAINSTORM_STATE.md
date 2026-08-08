# Architecture-brainstorm state (2026-08-06, session handoff doc)

Logan + Claude babble session on architecture/objective changes for the four-
ledger interpretability stack (see RESULTS_l0_mdl.md §33-49 for the bilin18
methods this serves). This file = the decisions and live queue so a successor
session can continue without the conversation.

## Queue (agreed order; E20/E21 launched)
1. E20 codebook slots (IN FLIGHT): n=256/k=2 VQ per slot on the E19a
   frontier-best arm (bandwidth 24x15 + lasso 1e-4). Registered: <=+0.15 CE
   promising, >+0.30 refutes granularity; dead codes <30%; PMI code-pair
   tables answer Logan's superposition-enumerability concern.
2. E21 predicate census (IN FLIGHT): bilin18 selection predicates scored on
   qk_e9_a + qk_e19_a heads, causal check on top-5/model. Seeds #3.
3. Predicate-basis attention: heads = learned mixture over named predicate
   library + lasso-priced free residual; iterative library growth (residual
   norm flags the next predicate to mine).
4. Identifiable wiring (Logan's read-scalar idea, sharpened): normalize each
   read group's columns to unit Frobenius, multiply by explicit scalar
   lambda_edge -> the 2,016-edge wiring table becomes literal parameters;
   lasso acts on lambdas; certified zero = lambda=0. Pure reparameterization,
   cheap, may improve wiring Spearman (removes gauge ambiguity).
5. Token-cone typed writes: per-module write = [token-span channel | free
   channel]; registered prediction: small CE cost bounded by free-share;
   deliverable = per-module "how much speech is token-like" + every-layer
   unembedding readability of the token channel. NOT expected to beat CE.
6. Unified differentiable bits (after E20): hard-concrete gates on read
   groups (wiring bits) + code-usage entropy (content bits) = one currency
   vs CE. Anneal in after warmup (strong-early kills channels — same
   dead-gradient trap as zero-init writes).
7. Declared-redundancy toy: task with test-time module-input corruption so
   redundancy is provably required; sweep copy count / mixing / ablation rate.
Parked: norms-free polynomial stack (RMSNorm folds exactly anyway), slot
decorrelation penalty (cov-composed metric already handles it), datastore
(most speculative; post-hoc substitution failed in memory-pipeline arc).

## Standing protocol decisions (this session)
- Two-tier size protocol: w128/6-block minutes-scale for MECHANISM smoke
  only; w264 minimum for any CE-frontier claim; w1152 scale spot-check
  (30-line CFG on scale box) before believing any structural win — the
  week's lesson: all four w264 structural wins flipped sign at w1152.
- Wiring-trajectory logging is now a standing harness upgrade: save the
  2,016-dim read-group-norm vector + per-slot census stats every 200 steps
  (few KB). Enables Logan's churn ground-truth test: validate that census-
  named parts (match heads, category MLPs) settle when the story says;
  if churn doesn't line up, try subspace-angle drift instead.
- Covariance-composed wiring Spearman is the standard reported metric
  (E17/E18; plain kept for continuity).
- Crystallization program shape (Logan-corrected): graft-and-grow, cheap
  automated harvests only (naming pass + predicate census + certified
  zeros), frozen pieces keep a trainable correction channel whose norm
  growth triggers re-dissolving. No full re-decomposition per iteration.

## Current frontier (w264 fresh, cov-composed Spearman)
vanilla 4.8513 | E19a bandwidth+1e-4: 4.9742 @ 0.826 (BEST tradeoff) |
E15c bandwidth 3e-5: 4.9038 @ 0.673 | E16b shrink+floor: 5.0231 @ 0.662 |
recipe E9a: 5.0547 @ 0.858. Scale: recipe combo3e5loss 4.106 (+0.141 over
Muon vanilla); all w264 structural wins flip at w1152; commons self-
organizes into token-channel + aggregation-bus roles (discovered not
imposed); typed commons + w1344 eff-param recipe in flight on scale box.

## Session mechanics for the successor
- Re-arm the autonomous cron tick (session-scoped, dies with the session);
  prompt template lives in the memory file qk-mdl-program.md tradition —
  current one covers: pull, read MAILBOX, check chains via exact-name pgrep,
  push verdicts, update chart artifact.
- Chart artifact URL (update via Artifact tool with url param):
  https://claude.ai/code/artifact/b5c81155-b26b-4a93-8368-b94adaf8ee01
- Fresh builder agents primed with AGENT_BRIEF.md; never resume long agents.
- Scale box coordination via MAILBOX.md (append-only newest-first) + git.

## Reviewer-2 findings (2026-08-06, pre-registration for successors)
R1 Spearman n=156 -> SE ~0.08: readability gaps <0.1 are TIES until
bootstrap CIs (from per-edge tables). R2 metric-Goodhart: validate wiring
Spearman once against real circuit-finding/edit success. R3 causal vector
is single-ablation on old cooc rows (off-distribution; ablations known
non-compositional) — recompute on fresh, spot-check pairwise. R4 single
seed/data-order: 2-3 seeds for retrain-recommendation arms. R5 token-cone
"span of embeddings" is VACUOUS (50k vecs span R^264) — only k-sparse or
convex-cone versions are real. R6 identifiable-wiring lambda degeneracy:
pattern path identifies only lambda_q*lambda_k — use one lambda per
(head,writer) on the product. R7 codebook: add distillation control
(expressivity vs trainability) + code dictionaries (else enumerable-not-
interpretable) + per-pursuit-step residuals. R8 census needs shuffled-
pattern nulls + z-scores + random-head causal controls. R9 churn confounded
by lr decay — normalize by update norms / constant-lr control.

## Standing logging requirements (the wish-we-had-logged list)
Per arm, always: per-edge consumption+weight tables; per-seq heldloss.npy;
wiring+slot-covariance snapshots every 200 steps + per-group update norms;
fixed audit slice (same ~200 seqs forever): patterns, slot contents, code
assignments; null rows (untrained-model Spearman, shuffled-pattern
predicate scores); codebook event logs (dead-code events, codebook
snapshots, pursuit residuals); step time + peak memory; seed + data-order
ids in JSON. One second-data-order replicate of the current best arm.

## Decision tree (interpretations conditional on results)
E20 codebook (cost vs parent E19a 4.9742):
- <=+0.15 & dead<30%: discrete content VIABLE -> proceed to unified-bits
  (#6) + inspect code dictionaries; sweep n/k only if dictionaries are
  semantically meaningful (else it's discrete-spectral, pause line).
- +0.15..0.30: run distillation control. Distilled << scratch = TRAINABILITY
  -> anneal-in quantization (soft->hard), retry. Distilled ~= scratch =
  GRANULARITY -> one retry at n=1024 or k=4. Still bad -> quantize only
  designated carriage slots (merges with k-sparse token-cone idea).
- >+0.30 with control confirming expressivity: discreteness at slot level
  REFUTED -> promote predicate/identifiable-wiring line to #1.
- Dead codes >30% but CE fine: shrink n; NOT a refutation.
- PMI pairs reused as stable units -> analysis unit = code PAIRS (Logan's
  superposition point confirmed; enumerate pairs). PMI ~independent ->
  codes compose freely (cleaner story).
E21 census (with shuffled nulls):
- Match/prev heads exist (z>3 + causal check passes): predicate-basis (#3)
  proceeds, library = the measured predicates, consider hardcoding them.
- NO nameable heads: check w1152 checkpoints (ask scale) before concluding;
  if absent there too, test whether recipe models do induction AT ALL
  (repeated-text advantage). Behavior present but no nameable head =
  distributed implementation -> predicate-basis MORE valuable (forces
  localization) but expect higher CE price; run it as a probe not a recipe.
- Only positional/key-class predicates: library starts there; match family
  may need architectural provision (idea 7).
Scale in-flight:
- Typed commons w1152: readability >= recipe & CE better -> typed commons
  enters retrain recipe. Readability drops like untyped -> typing does not
  transfer; commons line CLOSES; bandwidth+lasso is the only recipe core.
- w1344 eff-param-matched recipe: closes gap to Muon vanilla -> scale
  partition cost is mostly param deficit -> retrain = recipe + width bump.
  Doesn't close -> bandwidth reinvestment must be tested at scale.
- Bandwidth+1e-4 at w1152 (requested spot-check): HOLDS -> THE retrain
  core. FLIPS like the four structural wins -> w264 loses standing for
  structural claims entirely; pivot program weight to post-training
  (crystallization/distillation), which does not depend on width transfer.
Retrain recommendation rule: an architecture enters the recommendation only
with (a) CE better than recipe at readability-tie-or-better, (b) at BOTH
widths, (c) 2-3 seeds (R4). If nothing qualifies: recommend the confirmed
recipe as-is + post-training additions (anneal-to-certified-zeros,
cov-composed metric, census, naming pass) — those are width-independent.
Churn validation: churn tracks named parts -> crystallization proceeds with
churn triggers. Doesn't -> try subspace-angle drift. Neither -> naming
triggers stay manual (scheduled harvests), crystallization still viable.

## Component-specific customizations (per queued mechanism)
- Attention: pattern is a PRODUCT of reads -> quantization noise enters
  patterns quadratically, values linearly; report pattern-vs-value path
  degradation separately; fallback = quantize value-carried content only.
  Patterns can be NEGATIVE (no softmax) -> predicate library must be
  signed; fit positional predicates post-RoPE.
- Bilinear MLP: the natural FREE channel in token-cone typing (attention
  values = the token-cone candidates; MLPs compute). Left/Right reads are
  degree-2 in content (per-position only).
- Embedding: EXEMPT from VQ — it already IS a codebook (n=vocab, k=1);
  VQ = extending that discreteness to module channels. Token-cone
  (k-sparse form) = embedding as frozen codebook; tied embedding makes
  that channel exact-logit-readable at every layer.
- Per-slot RMSNorm CATCH: write-side gain dials are GAUGED AWAY (one
  writer/slot + reader-side renorm) -> dials are read-side only; idea 13
  merges into identifiable wiring. Codebook codes unit-norm + learned
  per-slot scale, else commitment loss fights the norm gauge.
- Readout: different interface (global pre-readout norm) — score/constrain
  there at global norm (E17/E18 showed top-10 0.9 at that interface); add
  per-block CE attribution in E20 (quantization cost may concentrate at
  the final block feeding the readout).
- Muon: matrices only. Codebooks (EMA, outside optimizer), edge lambdas,
  per-slot scales, predicate weights -> AdamW group. Identifiable wiring's
  unit-Frobenius read groups need re-projection after each Muon step
  (silent-drift bug otherwise).

## E21 census RESOLVED (2026-08-06): branch taken
Slotted heads are ~all POSITIONAL (recipe 8/72 >0.5, all positional-decay;
zero token-selective >0.5). Match family EXISTS but weak+distributed:
3-5 MATCH_prev heads/model at ~9% pattern mass, z~4400, causally confirmed
(joint beats profile-only >2 SE at 6/8 heads). No KEY-class heads (bilin18
had 30/162 programmatic). => Predicate-basis arm (#3) proceeds as a
LOCALIZATION PROBE, library = {signed positional profiles, MATCH_prev};
question = does an explicit named match term concentrate the distributed
component? Not a recipe candidate yet. Full tables qk_e21_census.json.
This is the successor's first build (E20 codebook verdict may land first —
read qk_e20.json and apply the decision tree above).

## E20 dictionary gate (2026-08-06): PASSED at token-class grade
Codes are coherent token/orthography detectors, same grade as the layer-0
QK atoms: mlp0 codes 180/123/172/248 = comma/'the'/period/'to' (thousands
of firings each); mlp7 code 120 = digits, 252 = subword continuation
(Random==>ised). Two distinct period codes in one slot (69 vs 193) = the
enumerable-superposition case — PMI follow-up. Meaning is token-class-
grade, not concept-grade (matches the program's content-is-token-shaped
record). => unified-bits objective (#6) UNGATED. w1152 branch point +
codebook spot-check running on the new scale box (s=65 solved, +35.4%
bandwidth, controls incl. tf32-symmetric identity passed).

## E20 error decomposition (2026-08-07, from logged heldloss + residuals)
Cost is TOKEN-heavy-tailed, document-uniform: worst 5% tokens = 80% of net
cost, 43% of tokens IMPROVE, worst-5% sequences = only 10%. Slot split:
mid/late ATTENTION slots ~45-46% content unexplained after 2 codes vs
MLP/early 24-25% — attention messages ~2x harder to quantize. => Queue adds:
E20b variable-k codebook (global budget, attention k=3-4, MLP k=2; predict
most of +0.134 collapses) and E24 code-to-code transition tables
(checkpoint-only: per-module input-codes->output-code contingency +
determinism measure — the discrete substitutability question, no training).
Pattern path stays continuous (quadratic sensitivity); hidden-layer
transcoder codebook and per-head codebooks queued behind.

## Open-problems ranking (2026-08-07, for compute allocation)
1 Content naming -> now a CATALOGING problem via codebooks (enumerate codes/
  pairs/dictionaries; LLM auto-labeling) — compute-solvable for enumeration.
2 Ablation non-composition -> BRUTE-FORCEABLE at w264: ~12K pairwise
  ablations = exhaustive 2nd-order interaction map, few GPU-h; replaces the
  first-order causal ground truth. High-priority compute buy.
3 MLP interior features -> hidden-layer transcoder codebook + capacity
  frontiers; code-to-code tables identify already-tabular MLPs.
4 Width transfer -> no predictive rule; NEW modification idea: saturation-
  gated mechanisms (strength scales with measured utilization — protective
  when starved, inert when not). Unbuilt.
5 Metric validity -> agent-based circuit-finding trials on high-vs-low
  Spearman checkpoints + bootstrap CIs. Pure evaluation compute.
6 Selection completeness -> predicate mining DSL; E22 residual norms.
7 Table enumeration -> extract/verify all memorized tables; ~100% with
  compute for bounded input sets.
Design rule (TN preservation): learned CONSTANTS fold (gates, lambdas,
mixture weights); learned FUNCTIONS of the input don't (routing/MoE) — keep
every new mechanism on the constant side.
Compute trio to fund first: pairwise-ablation map, metric-validation agent
trials, code-catalog enumeration.

## Methodological correction (2026-08-07, Logan): pattern signs need OV
Never interpret an attention pattern coefficient's sign alone in this
architecture. No softmax => pattern entries can be negative, and the OV
path can be negative too; negative x negative = net positive push on the
attended token's content (the negative effect lands on NON-matching
positions instead). The E22 "40/72 heads suppress via MATCH_prev" claim was
made on coefficient sign only and is retracted pending E28 (composed copy
score + causal b_h-zeroing on repeated text + 3-way confusion table).
Applies retroactively to any signed-kernel or signed-mixture reporting.

## FOUNDATIONS CORRECTION (2026-08-07): the readability axis is not yet a metric
E27 seed replicates: CE seed-stable (+0.019 / -0.012) but cov-composed
Spearman moved 0.128 on the recipe and the recipe-vs-frontier ORDERING
REVERSES with seed. => 3+ seeds mandatory for any readability claim; all
current readability rankings unsupported (R1's 0.08 tie threshold was, if
anything, too generous). Data order fixed => understates true spread.
E26 interaction map: only 18% of module pairs near-additive (predicted
>=70%); 148/300 superadditive; causal_ground_truth_changes_materially=TRUE.
=> the single-ablation vector every Spearman is scored against is
mis-specified. Fix: interaction-adjusted causal target.
E25 broadcast gates: priced PERMISSIONS select early detokenizers
(attn0/mlp0/attn1/mlp1...), priced MAGNITUDE (S2 write-lasso) selects late
aggregators (mlp11/mlp0/attn9/mlp10) — 1/4 overlap. Sharing "cast" is an
artifact of what you charge for, not a property of the model.
NEXT (highest priority, replaces prior queue order): (a) 3-seed protocol on
the 3-4 arms that matter; (b) rebuild the causal target with pairwise
interactions and re-score every stored wiring table; (c) only then revisit
frontier claims.

## The naming ceiling (2026-08-07, E32): stop adding predicates
After the 3 named terms absorb what they can, the learned residual pattern
is NOT nameable: 0/1/1 programmatic heads of 72 across 3 seeds (vs 42 in
the full pattern), nothing replicating across seeds, and unsupervised
structure diffuse (mean rank-1 mass 0.14, rank-4 0.33-0.35, 60-61/72 heads
diffuse). But the split is favourable: deleting the residual costs +0.44
nats and deleting the named terms costs +2.0-2.8, so the named library
carries 83-86% of total selection cost (induction 2.079 -> 1.755 residual-
deleted vs -> 0.307 names-deleted). CONCLUSION: iterative predicate-library
growth is AT ITS CEILING on this architecture; the residual 15% is a
genuine but unnameable remainder. Next moves should not be another named
term. Options that remain: (a) accept the 85/15 split and report it as the
architecture's honest coverage; (b) attack the remainder structurally
(different attention parameterization) rather than descriptively;
(c) spend effort on scale transfer + metric validity instead.
METHOD NOTE: within-sequence shuffle nulls give near-zero spread for DENSE
class features, so z-scores run in the thousands and are NOT effect sizes;
use joint-fit gain over the positional profile as the discriminating stat.

## BASELINE CORRECTION (2026-08-08): quote taxes vs MUON vanilla 4.7570
w264 taxes were being quoted against ADAMW vanilla (4.8513) while every
structured arm is Muon-trained. Muon vanilla is 4.7570 (qk_e0m.json), so
every tax was understated by 0.094. Corrected table (vs 4.7570):
predicate-basis +0.143 | composition +0.218 | bandwidth+dial +0.229 |
bandwidth gentle +0.147 | recipe +0.288 | codebook +0.352. Scale reporting
already used Muon vanilla, so this also removes a w264-vs-w1152
inconsistency (scale recipe premium +0.141 vs Muon vanilla is directly
comparable to w264 predicate-basis +0.143).
