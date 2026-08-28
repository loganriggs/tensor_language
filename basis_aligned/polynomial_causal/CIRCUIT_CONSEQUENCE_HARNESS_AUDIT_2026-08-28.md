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
