# Rung 465: paired necessity map of the shared equality correction

Status: prospective explanatory design, frozen after rung 464 and before opening any native/hybrid single-write
removal outcome. It uses the already-open 192-document code role. It does not fit a rank, sparse basis, or compressed
replacement.

## Why this is the next question

Rung 464 established a two-part operational circuit on code: native L8H4 and transplanted L5H5 equality scores are
interchangeable sources, and the complete set of later attention/MLP writes supplies an interchangeable context-
dependent correction. That grouped computation across native head boundaries, but the correction is still named as
one opaque set of 19 module outputs.

Rung 462 asked whether any one later write was **sufficient** by inserting it into a trajectory with no equality
source. Most such patches were harmful, which rung 464 now explains: the correction is not a standalone equality
signal. The complementary question was never tested: which later writes are **necessary when the source is present**?
This experiment removes the source-induced part of each write from a complete native or hybrid trajectory and lets
all later modules respond normally.

This directly advances within-module/cross-module circuit splitting, extraction, and selective removal. Rank,
storage, reconstruction, and aggregate CE are not discovery targets.

## Frozen intervention

Use the same pair, natural-fit scale, 192 code documents, six context cells, and fixed halves `0:96` / `96:192` as
rung 464. In each batch, cache the exact absent (`0`), native-L8H4 (`N`), and transplanted-L5H5 (`H`) trajectories
and their 19 public later writes:

`MLP8, attention9, MLP9, ..., attention17, MLP17`.

For each source `s` in `{N,H}` and each write `j`, rerun the complete source trajectory but replace only write `j`
with the same-document cached absent-trajectory write. Every later module after `j` then recomputes normally. This is
a one-site removal of the source-induced write, not insertion into an incompatible source-absent state and not a
frozen suffix.

MLP17 is the preregistered primary site because its source-present write inserted alone into the absent trajectory
was the largest rung-462 interferer (`-72.3%` recovery). All 19 sites are reported in their fixed order; no site is
selected to define the primary predictions after outcomes are seen.

## Computations

For source `s`, write `j`, and context cell `c`, first compute CE benefit relative to the absent trajectory:

`E[s,j_removed,c] = (sum CE[0,c] - sum CE[s,j_removed,c]) / token_count[c]`.

The signed necessity of the source-induced write is

`R[s,j,c] = E[s,full,c] - E[s,j_removed,c]`.

Positive `R` means removing the write loses useful equality behavior. Negative `R` means the write normally
suppresses or corrects an over-strong equality effect. For each site the four-context necessity vector is
`(near, far, one predecessor, multiple predecessors)`.

The total later-program correction for source `s` is inherited computationally from rung 464:

`K[s,c] = E[s,full,c] - E[s,direct_only,c]`.

Cosine between `R[N,j]` and `R[H,j]` asks whether the same site plays the same causal role for both interchangeable
sources. Cosine between `R[s,j]` and `K[s]` asks whether that site implements the full program's signed context
modulation. Vector norms are in nat across the four reported context effects; they prevent a nearly-zero vector from
passing by direction alone.

As a representation check, also accumulate within each site/context the cosine and RMS ratio between raw write
changes `write_N - write_0` and `write_H - write_0`. These quantities cannot select a site or pass a causal clause.

## Registered predictions

### A. Exact instrument

All parent/preregistration/model/row hashes hold; native replay error is at most `1e-12`; equality reconstruction
error is at most `1e-10`; absent/native/hybrid captured logits exactly reproduce rung-464 diagonal effects within
`1e-10 nat`; each patch has exact batch/position/module identity and fires once; all later modules execute after a
patch; the call census is exact; and SEALED remains closed.

### B. MLP17 has the same causal role for native and transplanted sources

The pooled four-context vectors `R[N,MLP17]` and `R[H,MLP17]` must each have norm at least `.02 nat`, cosine at least
`.80`, and larger/smaller norm ratio at most `2.0`. Their cosine must be positive in both fixed document halves.

### C. MLP17 carries the signed context correction, not merely generic loss

For both sources, MLP17's necessity vector must have the correction sign pattern—negative near and multiple-
predecessor entries, positive far and one-predecessor entries—pooled and in both halves. Its cosine with the full
later-program correction `K[s]` must be at least `.70` pooled and positive in both halves. Its four-context norm must
be at least twice the absolute off-target necessity and exceed it by `.01 nat` for both sources.

### D. The correction is a reproducible multi-site program across sources

Counting MLP17, at least two distinct sites must have pooled native/hybrid necessity cosine at least `.70` and
four-context norm at least `.01 nat` under both sources. Across all 19 sites, native versus hybrid necessity-norm
rankings must have Spearman correlation at least `.60` pooled and positive in both halves. This tests shared program
organization without assuming that native module boundaries are semantic atoms.

### E. The paired necessity map is not explained by raw write similarity alone

Across the 19 sites, causal-necessity norm and raw source-delta RMS must not be perfectly rank-equivalent: absolute
Spearman must be below `.95` for at least one source. In addition, MLP17 must rank in the top five causal-necessity
norms under both sources. This asks whether downstream use, rather than write size alone, locates the correction.

The strong null is an invalid instrument; MLP17 norm below `.005 nat` for both sources; pooled MLP17 native/hybrid
necessity cosine at most zero; no site with paired cosine above zero and norm at least `.005` under both sources; or
nonpositive complete-source all-positive benefit.

## Decision and claim boundary

- B/C pass: MLP17 is a shared necessary consumer/corrector of the cross-head equality variable and becomes the first
  fixed site for task-conditioned within-MLP splitting.
- D pass: multiple later modules form a source-invariant correction program; follow with a registered interaction
  factorial among the fixed necessary sites rather than adding their marginal effects.
- B/C fail but D passes: the correction is shared but MLP17 was only an incompatible-state sufficiency interferer;
  freeze the highest preregistered-criterion shared sites for later validation without rewriting this result.
- D fails: whole-program interchange exists, but its native and hybrid implementations distribute causal necessity
  differently; retain the source-plus-program abstraction and test a state-level rather than module-level boundary.
- E fails: raw write amplitude may already explain the site ordering; do not call the necessity map a new downstream-
  determined decomposition until a matched-amplitude control separates them.

Even a full pass is an already-open-code identification result with zero saved parameters. It does not by itself
establish a sparse MLP17 code, a scalar distance/count gate, fresh-corpus generalization, or adoption.
