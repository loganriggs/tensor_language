# L13H8 bracket-closure fixed-tensor canary v1

Status: **prospective, outcome-blind source scaffold; no execution authority exists**.

This is a discovery canary for one weight-fixed tensor edit. It cannot establish an
OOD circuit, a stack algorithm, a compressed model, or an MDL claim. A separate
authority/freezer/runner/result transaction is required before any forward.

## 1. Physical object and prohibited router

The target is squared-attention layer 13, head 8. Every nonnative arm owns all six
dense 1152-by-1152 projections, lambda, the 64 rotary constants, and one constant
nine-scalar projector on the head leg. The stored operator executes the native
RMS-normalized squared QK contraction, causal support, first-value bus, mixed values,
and output projection without calling native attention 13.

Delimiter parsing, stack depth, opener distance, code/prose type, target token,
family label, and every target/control mask are **score-only metadata**. They are not
passed to the replacement callback and may never choose a head, key, value, scalar,
or arm. No parser or stack receives executable credit.

## 2. Exact four arms

1. `native`: the unchanged model.
2. `stored_l13_all_heads`: exact stored replay of all nine L13 heads with projector
   `(1,1,1,1,1,1,1,1,1)`.
3. `stored_l13_delete_h8`: the same program and tensors with the constant projector
   `(1,1,1,1,1,1,1,1,0)`.
4. `stored_l13_deranged_h8`: full projector, but the 1152-by-128 H8 column slice of
   `c_proj` is replaced before authority by a materialized CPU-float64 SVD spectral
   derangement. The cyclic permutation is fixed-point-free, authority-hashed, and
   preserves the slice's singular values and rank. It has identical storage.

The null keeps H8's left/output singular directions and spectrum but deranges which
right/input singular direction is paired with each gain. Repeated singular values
make SVD coordinates noncanonical, so the materialized dense matrix—not merely the
recipe—is hashed. It is a weight-structure null, not a causal deletion.

Each stored arm costs exactly

`6 * 1152^2 + 1 + 64 + 9 = 7,962,698`

unquantized stored scalar values, has zero token-table values, total input support,
and zero native calls at attention 13. All other attention sites and all 18 MLPs are
native. Multiply-adds and stored values are reported separately. The full replay is
an autonomous stored extraction of L13 attention, not a compression.

## 3. Frozen roles and support

The only v1 roles are, in order:

- `select_prose`: natural prose documents;
- `select_code`: repository-Python documents disjoint by file and document from prose;
- `synthetic_canary`: generated strings, disjoint in row, document, and source identity.

All roles bind exact ordered `[N,257]` CPU-int64 rows, ordered document identities,
source-file identities, delimiter registry, score masks, and support hashes before a
forward. Pairwise row/document/source-file collision counts must be exactly zero.
Prediction columns are 0:256; registered natural inference uses columns 64:256.
There is no v1 OOD role and no authority may add one.

Natural delimiter families must contain at least round and square delimiters; braces
may be added only before authority and then become hash-bound. Opener, closer, quote,
and other-punctuation token-ID groups are disjoint. For a closer target, the CPU
prefix parser labels:

- compatible top-of-stack closer;
- incompatible-family closer with a nonempty stack;
- closer with no unmatched opener.

It also records closer family, pre-target stack depth, and distance to the unmatched
top opener. Compatible/incompatible natural cells are stratified by domain, family,
depth `1/2/3+`, and distance `1-8/9-32/33-128/129+`. Quote and non-delimiter
punctuation targets are typed controls. Compatible closers pop only when later used
as prefix tokens; malformed closers leave the score-only stack unchanged.

The synthetic role is balanced over domain, delimiter family, compatible versus
incompatible, all 3 depth bins, and all 4 distance bins, plus per-family no-opener and
per-domain quote/punctuation controls. It checks mask/parser support and qualitative
signs only; it is nonpromotive and never routes execution.

## 4. Currency and estimands

Every arm returns raw float32 logits `[documents,256,50304]` on identical rows. Row CE
is accumulated in float64. Teacher KL is `KL(p_native || p_arm)` on the same uncapped
logits; top-1 is secondary. For cell `c`, define document-balanced damage

`D_c(arm) = mean_document[ CE_c(arm) - CE_c(native) ]`,

with documents having no support omitted from that cell and their count reported.
The H8 necessity stake is `D_c(delete)`. Spectral-null recovered fraction is

`rho_c = (D_c(delete) - D_c(deranged)) / D_c(delete)`.

It is evaluated only when the simultaneous lower bound for `D_c(delete)` is positive.
Global collateral uses all common scored positions, not a target-selected denominator.

Inference uses one shared 20,000-draw source-document cluster bootstrap per natural
role, seed `2026083013`, and a replicatewise maximum absolute error over every
registered domain/family primary, control, and collateral coordinate. The 95% bound
is the zero-based order statistic 18,999 with no interpolation. No token bootstrap,
row bootstrap, normal-theory p-value, or post-hoc cell dropping is allowed. Empty or
nonfinite cells make the affected promotive conjunction unevaluable.

## 5. Discovery gates

All gates are conjunctive and descriptive until fresh confirmation exists:

1. **Integrity:** exact source/model/row/support/program hashes, role disjointness,
   finite logits, identical support, one outer forward per arm/batch, exact call
   ledger, zero native L13 calls for stored arms, and receipt-last publication.
2. **Replay:** on each natural role, stored replay versus native has maximum absolute
   logit error at most `1e-4` and pooled teacher KL at most `1e-8`.
3. **Necessity:** compatible-closer `D(delete)` has simultaneous LCB `> 0` separately
   in prose and code, and every authority-frozen delimiter family has point damage
   `> 0` in both domains.
4. **Specificity:** in each natural role the compatible-cell deletion damage minus
   the maximum deletion damage over incompatible, no-opener, quote, and punctuation
   controls has simultaneous LCB `> 0`.
5. **Collateral:** pooled all-position `D(delete)` has simultaneous UCB `<= 0.01 nat`
   in each natural role.
6. **Equal-price null:** wherever gate 3 is powered, spectral-null recovered fraction
   has simultaneous UCB `< 0.5`. Thus an arbitrary same-spectrum output orientation
   cannot receive more than half of H8's compatible-closer effect.
7. **Synthetic canary:** all registered cells have finite nonzero support; compatible
   deletion point damage is positive for every family/depth/distance/domain cell.
   Failure invalidates mask/support instrumentation but cannot promote the circuit.

Passing nominates a constant L13H8 physical deletion for a separately frozen,
file-disjoint natural OOD confirmation and later extraction/removal assay. It does not
show that H8 alone computes a bracket stack, that the parser is mechanistic, or that
the effect is selective outside the registered cells. Failing replay invalidates the
backend. Failing necessity prunes this canary. Passing necessity but failing
specificity indicates broad punctuation/domain use. Failing the equal-price null
means weight orientation is not identified by this assay.

## 6. Source and lifecycle boundary

The exact source closure is frozen by `bracket_closure_canary_v1.SOURCE_CLOSURE` and
must be committed and pushed before authority. Authority must bind exact config and
checkpoint files, all role and support artifacts, exact materialized programs, and
the derangement. Source/model/role/program hashes are rechecked before every role and
immediately before payload, manifest, and receipt publication. A failure is terminal,
preserved, and cannot publish a result receipt. These source files themselves provide
no loader, freezer, CLI, forward loop, result scorer, or publisher, so current status
remains execution **NO-GO**.
