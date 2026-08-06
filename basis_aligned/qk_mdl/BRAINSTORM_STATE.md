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
