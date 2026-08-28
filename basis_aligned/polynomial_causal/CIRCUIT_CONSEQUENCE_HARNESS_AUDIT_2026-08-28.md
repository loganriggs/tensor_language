# Audit of the bootstrap circuit-consequence harness

Status: static design audit of `basis_aligned/bilinear_quotient/ops/circuit_audit.py`
at commit `7a580264`. This document does not authorize a new data role or promote a
scientific result.

## What the current harness validly measures

For a hand-written set of component groups, the harness measures three descriptive
quantities on positions whose current token appeared in one fit cache:

1. the CE increase when every component in the group is replaced by a stored constant;
2. the fraction of that increase recovered by independently fitted current-token mean
   output tables;
3. replication of those quantities from FineWeb `skip7000` to FineWeb `skip11000`.

That is a useful, inexpensive **component-set screen**. It can identify groups whose
generic document-level behavior is important and unusually token-tableable. It cannot
yet support the stronger circuit-consequence labels in its header.

## Required claim corrections

| current label | actual estimand | required stronger test |
|---|---|---|
| `REMOVAL` | importance of a whole component set under constant replacement | a declared circuit variable is edited or removed; target behavior changes as predicted; behavior outside declared descendants stays within a preregistered collateral bound |
| `EXTRACTION` | causal replacement by one current-token table grammar | compare multiple frozen program grammars at matched causal distortion; require zero native calls and score target plus off-target behavior |
| `OOD` | held-out replication between two FineWeb document offsets | freeze selection and fit before a genuinely shifted corpus, code distribution, synthetic counterfactual, or declared trigger-composition split |
| `circuit` | registry entry mapped to one or more whole native modules | an executable typed subprogram with declared inputs, outputs, trigger support, causal descendants, and intervention semantics |

Several registry entries currently map to the same native component set. They must
receive identical outputs even when their semantic claims differ. This is not evidence
that the semantic circuits behave identically; it is evidence that the instrument has
not represented the difference.

## Why this does not yet validate a simplicity definition

The project-level criterion is prospective consequence prediction:

$$
C_j(P)<C_j(Q),\qquad D_{\mathrm{val}}(P)\simeq D_{\mathrm{val}}(Q)
\quad\Longrightarrow\quad Y_j(P)>Y_j(Q).
$$

The present harness evaluates one nonconstant grammar—the token table—and never
computes competing simplicity measures over a common candidate bank. It therefore
cannot tell whether parameter count, gauge-quotiented dimension, MDL bits, typed-graph
sparsity, rank, degree, or causal-interface size predicts extraction, removal, or OOD
better. It supplies possible outcome columns for such a comparison, not the comparison
itself.

## Minimum consequence-test schema

Every row in the upgraded harness should bind:

- `circuit_id` and immutable source claim;
- executable `program_id`, grammar, artifact hash, and zero-native-call receipt;
- declared causal inputs, outputs, trigger mask, descendants, and off-target support;
- complexity vector rather than one scalar: serialized bits, structural degrees of
  freedom after gauge, FLOPs/depth, typed nodes/edges, interface dimension, rank/degree;
- causal distortion on a validation role used only to form matched-fidelity pairs;
- final held-out target effect, collateral effect, composition-prediction error,
  extraction recovery, latency/memory, and OOD effect;
- role provenance proving that final consequence data did not choose the program,
  simplicity definition, threshold, trigger mask, or circuit mapping.

The first useful statistical unit is not “one score per registry prose entry.” It is
a paired comparison between two executable candidates for the same declared causal
interface and approximately the same validation distortion. Definitions are evaluated
by held-out pairwise accuracy and rank correlation, with entire circuits or grammar
families held out during cross-validation.

## Immediate implementation gates

Before running the bootstrap repeatedly:

1. add a real CLI with `--help`, `--dry-run`, and `--output`; currently `--help`
   launches the full GPU experiment;
2. refuse an occupied GPU or integrate with the repository runner/ownership protocol;
3. cache fitted tables by the canonical component-set tuple—many registry entries
   duplicate the same set and the current code refits it once per entry per eval split;
4. name the existing outputs `component_constant_ablation`,
   `token_table_recovery`, and `fineweb_split_replication` in the artifact, retaining
   backward aliases only if necessary;
5. report trigger coverage and target/off-target cells; a generic CE average can make
   a narrow, correct circuit look inert;
6. add source commit, registry hash, row hashes, checkpoint identity, component-map
   hash, and overwrite protection to every receipt;
7. handle near-zero removal denominators with a frozen absolute floor and confidence
   interval rather than an unstable relative OOD ratio.

Until these gates land, the harness should remain explicitly labeled **bootstrap
descriptive screening**. The governing definition-to-consequence protocol remains
`SIMPLICITY_CONSEQUENCE_VALIDATION_V1.md`.

## First-run evidence

The first committed run at `d1e97c41` completed in 144.7 seconds and audited 16 of
55 certified entries. Its known controls held, and current-token table recovery spans
approximately $-19.3\%$ to $96.1\%$. The strongest positive rows are MLP1 ($96.06\%$)
and MLP0 ($91.01\%$); all-attention and middle-attention tables are negative
($-1.88\%$ and $-19.34\%$). This is coherent descriptive evidence that early MLP
writes are unusually token-local while attention writes require cross-position state.

It does not remove the claim-boundary problem. Three different registry claims about
the middle band receive bit-identical scores because all three resolve to MLP4--15.
The run therefore validates the screen as a discriminator between component-set
replacement grammars, not as an evaluator of semantic circuits or simplicity
definitions.

## v2 matched-size control audit

`circuit_audit_v2.py` adds one deterministic same-size component-set ablation and
reports named-set removal divided by control-set removal as `specificity`. This is a
useful relative-importance diagnostic, and its first run correctly fails the registered
claim that the median named set beats its selected control. It still is not circuit
specificity or collateral control.

The ratio is strongly identified by the one chosen denominator. For the single-site
rows, MLP0 is controlled by the exceptionally consequential MLP1 and MLP1 is controlled
by MLP0, producing nearly reciprocal scores $0.1213$ and $8.2473$. Front MLPs are
compared with a deterministic spread of later MLPs; all MLPs are compared with all
attention, changing component kind. These comparisons mostly restate depth and module
importance. Duplicate registry claims also repeat identical ratios and cannot be used
as independent observations when taking a median.

The field should therefore be interpreted as **single-control relative component-set
importance**. A stronger module-set null would enumerate or sample many same-kind,
same-cardinality sets, stratify or match depth and baseline stake, report the null
distribution and percentile, deduplicate canonical component sets, and avoid ratios
whose numerator or denominator is CE-saturated. Even that would not measure selective
removal: circuit specificity still requires trigger-target and trigger-off-target cells
with collateral bounds on declared non-descendants.

## v3 claim-direction audit

The v3 multi-draw percentile repairs the most severe single-denominator artifact. Its
hand-written `important` / `redundant` direction, however, is not yet a claim predicate.
Most registered entries do not assert that their component set has unusually high or
low global ablation cost:

| registry claim family | estimand actually required |
|---|---|
| MLP0/MLP1 dossiers | token-local versus context-conditioned recovery plus declared semantic interventions |
| front tableability / front-versus-middle | depth gradient or contrast in matched-grammar extraction recovery |
| middle program-family / feature price curve | matched-fidelity complexity frontier and held-out marginal return |
| front or band synergy | factorial or Möbius interaction, not marginal ablation magnitude |
| attention nonlocal / two-position / lag failure | token-only versus lag/context-aware grammar contrast |
| whole-model / best compiled program | zero-native-call whole-program CE, KL, composition, and OOD consequences |
| routing-only compressibility | separately priced routing-versus-value replacement frontier |

Consequently, annotating `_attention_write_is_mostly_two_position` as `important`
does not make a removal percentile evidence for a two-position mechanism; likewise a
high removal percentile cannot test synergy or a price curve. v3 should be read as a
better **component-importance null**, with direction annotations as metadata. A future
harness must dispatch each typed claim to its required estimand and mark unsupported
claim/metric pairs unauditable rather than collapsing them to importance.

### v3 first-run outcome and statistical-unit correction

The v3 run completed in 239.5 seconds. Its multi-draw null does repair the v2 artifact:
MLP0 moves from a single-control ratio of $0.12$ to $14.81$ against the control median,
MLP1 moves from $8.25$ to $151.56$, and both land at percentile $1.00$ among twelve
draws. The registered discrimination, self-diagnosis, and replay controls pass.

The registered direction score also passes as written at 5/7, but registry entries are
the wrong statistical unit. `_front_band_tableability_ladder`,
`_front_is_tabular_middle_is_not`, and `_front_mlps_are_synergistic` all map to the
same MLP0--3 set, receive the same percentile $1.00$, carry the same `important`
direction, and contribute three successes. Deduplicating by canonical component set
plus direction gives:

- MLP0: important, pass;
- MLP1: important, pass;
- MLP0--3: important, pass;
- MLP4--15: redundant, fail at percentile $0.33$ against the $\leq0.25$ bar;
- attention4--15: important, fail at percentile $0.00$.

That is 3/5 = 60%, below the registered two-thirds threshold. This does not rescore or
erase v3's preregistered 5/7 pass; it shows that the pass is not evidence that claim
direction repairs the harness. The next evaluation must declare its independent unit
before scoring and must not treat multiple prose claims sharing one measured object as
replicates.

## v4 causal-mask failure and claim boundary

The first v4 run and its `class_ratio_site_sweep` follow-on do **not** measure the
documented induction class. In the mask matrix, rows are current positions $j$ and
columns are candidate source positions $p$, but the implementation used

$$
j < p
$$

while its comment and scientific interpretation asserted $p<j$. It therefore searched
future bigram matches. The repeat mask used the correct past-facing inequality, so the
three labels remained disjoint and exhaustive and the count-sum control could not catch
the error. A four-token known-answer case makes the failure explicit: for
`[5, 7, 5, 7]`, position 2 is the valid repeat of the earlier transition $5\to7$;
the original mask instead labels position 0 using the matching transition in its future.

On all 192 `skip7000` rows before fit-token coverage filtering, the correction changes
4,564 of 36,864 scored-position induction labels. Counts move from
2,864/15,194/18,806 to 4,166/13,010/19,688 for induction/repeat/novel. Therefore the
v4 class-conditioned ratios, its §1727 interpretation, the completed held-out site
sweep, and the follow-on joint-ratio interval are invalid. Their artifacts remain failure evidence and must not be silently
overwritten or interpreted. The corrected implementation lives in
`ops/target_token_classes.py` with tests for past/future asymmetry, suffix invariance,
and an exact partition. A corrected v4 replay is descriptive because the predictions
and follow-on were already exposed; any confirmation requires a newly frozen hypothesis
and untouched data role.

Even with the mask corrected, these token classes are only a generic target-side
stratification. They are not automatically a circuit's trigger set, off-target set, or
non-descendants. Consequently, claimed-class/complement damage is a useful stratified
ablation profile but is not yet selective circuit removal or collateral damage. Those
labels require a typed circuit-specific trigger, intended output, declared causal
descendants, and off-target support, frozen before intervention.

### Corrected descriptive replay

The corrected v4 replay completed in 51.2 seconds and wrote a separate artifact rather
than overwriting v4. On fit-covered `skip7000` positions, class counts change from the
invalid 2,341/10,885/14,748 to 3,394/9,127/15,453. The numerical interpretation changes
materially: middle-attention removal is 2.664 nats/token on induction targets versus
2.248 on novel targets, and its claimed-class/complement selectivity moves from 1.060
to 1.308. The all-attention induction/novel damage ratio is 1.165 versus 0.881 for all
MLPs. The old four Boolean predicates move from false/true/true/false to all true.

These values are a corrected **discovery profile**, not restored preregistered results.
Both original eval roles were exposed, and the predicates were observed under the old
mask before correction. The next legitimate confirmation must freeze a hypothesis from
the corrected profile and evaluate a newly declared role. It must still avoid calling
generic target-class selectivity selective circuit removal.
