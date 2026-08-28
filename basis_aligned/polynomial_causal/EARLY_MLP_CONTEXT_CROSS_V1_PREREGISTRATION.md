# Early-MLP/context tensor-cross v1 preregistration

**Frozen:** 2026-08-28, before any model outcome on this registry

**Status:** frozen statistical design, **NO-GO for model execution** until the
source-closed runner/lifecycle amendment below passes independent review; no
final-role, semantic, OOD, edit, storage, or whole-model ledger authority

**Executable registry:** `early_mlp_context_cross_v1.py`

## Question

Can the nonadditive effect of substituting MLP0/MLP1/MLP2 and downstream contextual
sites be predicted through a rank-three or rank-four interaction state selected
without observing the new grid?

This follows four completed facts:

1. exact MLP2 restoration changes sign depending on whether MLP0 and MLP1 have been
   restored;
2. independent per-site table choices become worse when composed;
3. a layer-5 rank-at-most-two completion failed on untouched interactions; and
4. the completed layer-5 CE grid has a stable rank-four maximum-volume pivot under
   2,000 document bootstraps.

The old layer-5 outcomes are used only to freeze the **ordinal nested cross pattern**.
No old cost enters a fit or score here.

## Physical program and currency

For cell \((i,j)\), replace exactly the sites in

\[
P_i\cup S_j
\]

with the committed section-1786 program family: context-free covered rows with
centered rank-64 truncation, a rank-64 embedding-to-row map executed for uncovered
tokens, identity gains, and no mask-specific refitting. Output-distribution
nearest-neighbour indices are a hashed control only; they are **not** the executed
uncovered-token path. All other sites remain native. The backend executes a native
module before substituting its output, so this assay establishes behavioral
factorization, not zero-native-call execution cost.

Unlike compilation-mask cut-rank v1, the \((0,0)\) anchor is the fully live model:
there are no always-compiled sites. MLP0 is therefore a genuine experimental factor.

Primary cost is CE increase in nats relative to live on identical scored positions:

\[
H^{\mathrm{CE}}_{ij}=\mathrm{CE}(P_i\cup S_j)-\mathrm{CE}(\varnothing).
\]

Mandatory secondary cost is top-1 accuracy loss in percentage points. CE alone
selects the cross because its old rank-four pivot was stable in 1,999/2,000 document
bootstraps, whereas the top-1 pivot was stable in only 860/2,000. Top-1 and all
future causal coordinates are outcomes, never selection currencies.

## Frozen mask registry

The physical cut is after layer 2. Prefix masks contain only early MLPs:

| index | prefix mask |
|---:|---|
| 0 | empty |
| 1 | MLP0 |
| 2 | MLP0 + MLP1 |
| 3 | MLP1 |
| 4 | MLP0 + MLP1 + MLP2 |
| 5 | MLP2 |
| 6 | MLP1 + MLP2 |
| 7 | MLP0 + MLP2 |

Suffix masks contain only layers 3--17:

| index | suffix mask |
|---:|---|
| 0 | empty |
| 1 | attention3 |
| 2 | MLP3 |
| 3 | attention3 + MLP3 |
| 4 | all attention at layers 3--8 |
| 5 | all attention and MLP sites at layers 3--8 |
| 6 | all attention at layers 3--17 |
| 7 | all attention and MLP sites at layers 3--17 |

The non-monotone row ordering is intentional. It leaves singleton MLP0, pair
MLP0+MLP1, and triple MLP0+MLP1+MLP2 in final heldout rows while still measuring
single and paired early-MLP effects in the cross.

## Frozen nested crosses and staged evidence

Remove additive row/column effects:

\[
\Delta_{ij}=H_{ij}-H_{i0}-H_{0j}+H_{00}.
\]

Rank-three uses rows \(\{3,6,7\}\) and columns \(\{2,4,7\}\). Rank-four uses
rows \(\{3,5,6,7\}\) and columns \(\{2,4,6,7\}\). The former is a strict subset
of the latter.

- **Rank-three discovery:** all 15 anchors plus its 33 inner cross entries, 48
  cells total.
- **Rank-three validation / rank-four expansion:** the seven entries present in the
  rank-four cross but absent from rank three.
- **Rank-four final heldout:** the nine Cartesian cells with prefix rows
  \(\{1,2,4\}\) and suffix columns \(\{1,3,5\}\).

For rank \(r\), with pivot \(U=\Delta[I_r,J_r]\), predict

\[
\widehat\Delta=\Delta[:,J_r]U^{-1}\Delta[I_r,:].
\]

No adaptive replacement pivot is allowed. If a frozen pivot is singular or misses
its conditioning gate, that rank fails.

The measurement backend may collect all cells in one source-closed transaction, but
the scorer must enforce staged access through exact cell capabilities: rank three
accepts exactly its 48 discovery cells; rank four accepts exactly its 55 fit cells;
either rejects any extra cell before inspecting its value. Validation and heldout
scorers are dedicated stages that accept exactly their seven or nine licensed cells.
Caller-selected score subsets are forbidden. Forbidden finite and NaN cells must
both fail closed.

## Rows, replication, and uncertainty

Use two document-disjoint, already cached 192-row roles (`skip7000` and `skip11000`)
only after a receipt proves their ordered document sets are disjoint. Each contributes
192 scored targets per row. Retain per-document correct counts and CE sums for every
cell. No pooling across roles is allowed for a pass.

CE and correct-count costs are token-weighted: sum per-document CE/correct counts and
divide by the sum of per-document scored-token counts, both for the point estimate
and inside each document bootstrap. Equal-document averaging is forbidden.

Use 2,000 source-document bootstrap resamples per role. The exact seeds are
`2026082803` for `skip7000` and `2026082804` for `skip11000`. Recompute cost grids,
pivots, predictions, and metrics inside each draw. Report literal 2.5%/97.5%
percentile intervals and pivot-condition quantiles.

For a role with \(D\) source documents, draw exactly \(D\) document indices with
replacement per bootstrap draw and convert them to one multiplicity vector. The same
vector is applied to every one of the 64 cells, CE and top-1, cross and ALS baselines,
and every subgroup statistic in that draw. The token denominator is recomputed from
the multiplicities; it is not held fixed. Quantiles use linear interpolation between
order statistics (Hyndman--Fan type 7, NumPy `method="linear"`). No singular draw is
dropped: any singular frozen pivot makes that role/rank fail its bootstrap gates.
Quantiles over the remaining nonsingular draws may be reported descriptively with
the singular count, but cannot rescue the failed conjunction.

These roles have appeared in prior experiments, so this is a prospective **new-mask
outcome** test and a cross-role replication, not fresh-corpus OOD evidence.

## Baselines and metrics

For the same validation or heldout cells, report:

- cross RMSE and maximum absolute error;
- heldout total-cost \(R^2\);
- interaction NRE on the scored cell set,
  \(\|\widehat\Delta_C-\Delta_C\|_2/\|\Delta_C\|_2\);
- additive-anchor error, which sets \(\widehat\Delta=0\);
- cross RMSE divided by additive RMSE; and
- a same-cross-entry rank-\(r\) ridge-ALS baseline with eight frozen restarts,
  seed `2026082805`, and no validation/heldout tuning. It fits
  \(L\in\mathbb R^{7\times r},R\in\mathbb R^{r\times7}\) only to the same observed
  anchored-interaction cross entries. First divide all observed interactions by
  their root-mean-square; zero RMS is a failed baseline. In those dimensionless
  coordinates minimize mean observed-entry squared error plus
  \(10^{-6}(\operatorname{mean}L^2+\operatorname{mean}R^2)\), then multiply the
  prediction by the original RMS. There is no penalty grid.

  For restart \(k\), initialize both factors iid normal with standard deviation
  \(r^{-1/2}\) from a fresh generator seeded `2026082805 + 1000*r + k`. Run exactly
  100 sweeps, updating all rows of \(L\) in ascending order and then all columns of
  \(R\) in ascending order by their closed-form ridge solutions. Select the smallest
  final penalized objective, breaking exact ties by the lower restart index. Refit
  from scratch inside every bootstrap draw using the same restart seeds. This
  normalized rule is invariant to multiplying the entire interaction grid by a
  nonzero scalar.

The rank-four representation pays for 15 anchors plus
\(14r-r^2=40\) interaction entries, 55 values versus 64 for the complete local
grid. This modest local saving is not itself the goal; passing licenses cross
selection across several physical cuts, where tensor-train measurement cost can
scale linearly with depth.

## Registered gates

### Rank-three validation gate

On the seven validation cells, separately on both roles:

1. point pivot condition number is at most 20 and bootstrap 95th percentile at most
   25;
2. CE interaction NRE is at most 0.50 and its bootstrap upper bound at most 0.65;
3. CE RMSE/additive-RMSE is at most 0.75 and its bootstrap upper bound at most 0.90;
4. CE total-cost \(R^2\) is positive with bootstrap lower bound above zero; and
5. point CE cross RMSE is no worse than point same-entry rank-three ridge ALS.

If every gate passes, rank three is the selected minimal model. Rank four is still
scored on its frozen heldout cells but cannot replace a passing rank three as the
simplicity winner.

### Rank-four final gate

On the nine heldout cells, separately on both roles:

1. point pivot condition number is at most 20 and bootstrap 95th percentile at most
   25;
2. CE interaction NRE is at most 0.50 and its bootstrap upper bound at most 0.65;
3. CE RMSE/additive-RMSE is at most 0.75 and its bootstrap upper bound at most 0.90;
4. CE RMSE is at most 0.10 nats and its bootstrap upper bound at most 0.15 nats;
5. CE total-cost \(R^2\) is at least 0.50 with bootstrap lower bound above zero;
6. point CE cross RMSE is no worse than point same-entry rank-four ridge ALS; and
7. cross RMSE is no worse than additive separately for each heldout prefix order
   (singleton, pair, triple) and suffix class (local attention, local block,
   shallow dense).

Top-1 must report identical metrics and subgroup results but is not part of model
selection or the CE useful-pass conjunction. A top-1 failure prevents a broad
behavioral claim, not the narrower CE factorization claim.

## Interpretation

- **Rank three passes:** a 48-cell nested cross predicts its seven untouched
  expansion cells on two roles. If rank four also passes final, rank three is the
  simpler selected local state.
- **Rank three fails, rank four passes:** the registered rank-four predictor is
  adequate while the registered rank-three predictor is not. This does not prove
  intrinsic dimension at least four, because fixed-pivot conditioning and threshold
  placement are alternative explanations.
- **Rank four fails:** prune low-rank cross interpolation for this early-MLP/context
  mask family. Do not search a post-outcome pivot or increase rank on these heldout
  cells.
- **Either passes:** this licenses an adjacent-cut/vector-response assay only. It
  does not establish global TT rank, semantic features, OOD transport, selective
  editability, zero-native-call execution, or whole-model compression.

## Frozen parent evidence

- tensor-cross bootstrap result SHA256:
  `3951cd55a1e767e3212791622463057abe86601c5dca1ec30eff3a1a89e247fd`
- parent measurement payload SHA256:
  `04fc912cbce96d0a07564c1984565996b35a86dd1279bc00b45f5e842d8f9d75`
- parent bootstrap code SHA256:
  `2535a58d13cbe520f978a2f44d0934a02234aa61b3a5adf3fbcfb5cfcf74096c`
- parent source commit: `4894395b`

Any implementation amendment must be committed and pushed before model outcomes,
must preserve these masks/cells/gates, and must enumerate any operational change
without using a spent result namespace.

## Launch blockers and required lifecycle amendment

This file does not authorize a GPU/model run. Before execution, a separately reviewed
amendment must pin the exact two row tensors and their provenance, the frozen model
realization, the immutable section-1786 program bank, all source hashes, and a fresh
create-only namespace. It must implement the two-role per-document collector,
capability-separated discovery/validation/heldout scoring, the fixed ALS/bootstrap
rules above, pre-load and post-load hashes, a lock, failure receipt, last-write result
receipt, and no publication before terminal closure.

Existing reusable inputs, to be revalidated rather than silently trusted, are the
192-row cache files with serialized SHA256 `d66c1ee7...` (`skip7000`) and
`b1564bfd...` (`skip11000`). The prior source receipt is `815b2161...`; the roles
contain 79 and 105 source documents with zero overlap. The committed backend family
is `section1786_contextfree_rank64_table_learned_rank64_map`; current backend source
SHA256 is `738f4988...`. These abbreviated hashes are descriptive only: the launch
amendment must store and verify full hashes.
