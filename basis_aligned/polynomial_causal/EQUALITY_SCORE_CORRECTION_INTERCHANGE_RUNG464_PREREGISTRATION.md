# Rung 464: native-versus-transplanted matcher correction interchange

Status: prospective explanatory design, frozen after rung 463 and before opening any native/hybrid cross-pair
outcome. It uses the already-open 192-document code role. This is a circuit-interaction test, not a rank,
compression, or new-corpus confirmation result.

## Question

Rung 463 crossed the native L8H4 equality signal with the complete set of later attention/MLP writes. The direct
signal alone recovered 128.4% of the native all-positive effect, but it over-restored near and multiple-predecessor
cases and changed the off-target cell by `-.0254 nat`. The later writes alone were harmful overall, yet their signed
effect depended on context: they suppressed near and multiple-predecessor cases and helped far and one-predecessor
cases. The intact model therefore looks like a strong equality signal followed by a distributed context-dependent
correction.

Rung 459 showed that L5H5's complete double-QK equality score can drive L8H4's value/output branch. Rung 461 showed
that this transplanted matcher reproduces the native context ordering on code. The remaining circuit question is
whether native and transplanted matchers are treated as the **same downstream variable**, including the later
correction, or whether each matcher only works with the exact later trajectory it induced.

## Frozen trajectories and 3 x 3 intervention

For each same-document batch, cache three complete trajectories and all 19 later public writes (`MLP8`, then
`attention9`, `MLP9`, ..., `attention17`, `MLP17`):

- `0` (absent): L5H5 and L8H4 equality terms both removed;
- `N` (native): L5H5 removed and native L8H4 equality restored;
- `H` (hybrid): both native terms removed, then L5H5's frozen natural-fit equality score is contracted with L8H4's
  native value/output payload.

Run all nine combinations `(source, later-write donor)` in `{0,N,H} x {0,N,H}`. The source chooses the equality
write at attention8. The donor chooses the complete cached set of 19 later attention/MLP residual writes. Thus
`(0,0)`, `(N,N)`, and `(H,H)` replay the three intact trajectories; `(N,0)` and `(H,0)` expose each direct signal;
`(0,N)` and `(0,H)` insert each correction program without a signal; and `(N,H)` / `(H,N)` are the two critical
crosses. Attention patches replace only the public residual write and keep the recipient trajectory's current
first-value channel.

Documents, masks, frozen halves `0:96` / `96:192`, pair, scale, factors, and cells are inherited without search
from rungs 461--463. No QK branch, reader, scale, module subset, row role, or threshold may be selected after seeing
the result. SEALED attention0 outcomes remain closed.

## Computations

For context cell `c`, define the CE benefit of a combination relative to the absent trajectory as

`E[s,w,c] = (sum CE[0,0,c] - sum CE[s,w,c]) / token_count[c]`.

Lower CE is better, so positive `E` means the intervention restores useful computation. The later correction added
to source `s` by donor `w` is

`K[s,w,c] = E[s,w,c] - E[s,0,c]`.

The four-cell correction vector is ordered as `(near, far, one predecessor, multiple predecessors)`. A correction
program has the rung-463 context pattern when its near and multiple entries are negative while its far and one-
predecessor entries are positive. Vector cosine measures whether two programs make the same signed context changes;
the norm ratio checks that agreement is not obtained by shrinking one program to nearly zero.

For a crossed intact circuit, recovery is its all-positive benefit divided by the corresponding matched benefit:

- native source with hybrid correction: `E[N,H,all] / E[N,N,all]`;
- hybrid source with native correction: `E[H,N,all] / E[H,H,all]`.

## Registered predictions

### A. Exact instrument

All parent/preregistration/model/row hashes hold; native replay relative-squared error is at most `1e-12`; equality
factor reconstruction error is at most `1e-10`; every cached write has exact batch/position/module identity; every
declared patch fires exactly once; `(0,0)`, `(N,N)`, and `(H,H)` reproduce their directly captured logits to relative-
squared error at most `1e-12`; the call census is exact; and SEALED remains closed.

### B. Native and hybrid trajectories induce the same kind of context correction

`K[N,N]` and `K[H,H]` must both have the rung-463 four-cell sign pattern pooled and in both fixed halves. Their
four-cell cosine must be at least `.80` pooled and positive in both halves. Their norm ratio, always larger norm over
smaller norm, must be at most `2.0` pooled.

### C. The two correction programs are interchangeable across matcher implementations

For native source, `K[N,H]` versus `K[N,N]`, and for hybrid source, `K[H,N]` versus `K[H,H]`, must each have cosine
at least `.80` pooled and positive in both halves. Each crossed correction norm divided by its matched correction
norm must lie in `[.50, 1.50]` pooled. This is the direct test that downstream layers treat the matcher as one
variable rather than as two source-specific coordinates.

### D. Crossed complete circuits preserve the useful equality computation

Both crossed all-positive recoveries defined above must be at least `.75` pooled and positive in both halves. Both
crosses must preserve `far > near` and `one predecessor > multiple predecessors` for their CE benefits pooled and in
both halves. Each crossed circuit's absolute off-target benefit must be at most `.01 nat`.

### E. A correction program is not a useful standalone circuit

For each donor `w` in `{N,H}`, the correction-only arm `(0,w)` must have nonpositive all-positive benefit or negative
benefit in both near and multiple-predecessor cells, pooled and with the near/multiple signs stable in both halves.
This control distinguishes a context-dependent correction from a second independent equality signal.

The strong null is an invalid instrument; nonpositive all-positive benefit for either matched intact circuit; pooled
matched-correction cosine at most zero; either crossed recovery at most `.25`; or failure of both crossed circuits
to preserve either pooled context ordering.

## Decision and claim boundary

- B--D pass: native L8H4 and transplanted L5H5 scores are one downstream equality variable at source-plus-correction
  circuit grain. The shared component may cross native head boundaries, while the distributed later correction is a
  reusable consumer program.
- B passes but C/D fail: the two matchers have similar aggregate context effects but are conditionally coupled to
  source-specific later responses; the whole score is not yet an interchangeable circuit component.
- B fails: the apparent shared matcher does not induce a stable common correction program; retain the narrower
  score/payload screen and diagnose which later context feature differs.
- E fails while B--D pass: the correction itself carries equality information and should be decomposed as a second
  source rather than described only as a correction.

Even a full pass is identification on an already-open code role, not a saved-parameter model or a new OOD
confirmation. It changes the cross-module grouping, interaction, extraction, and selective-interchange circuit
targets. Rank and storage remain zero-weight diagnostics here.
