# Rung 439 preregistration — causal-profile-scored Archetypal Q/K hull

Date: 2026-09-01T22:00Z

Claim level: structural identifiability screen. A pass does not adopt a replacement or assign semantic names.

## Decision

Rung430 established a useful score-trained global sparse Q/K generator, but independent restarts recovered unstable
private decoder atoms. Test whether anchoring those atoms to real token geometry improves *observable* identifiability
without sacrificing the computation. This is the first Archetypal-SAE test; it is not a rank, active-count, or
precision sweep.

## Exact objects

For query side `s=q` and key side `s=k`, let `X_s(t) in R^2304` concatenate all 18 layer-0
`(head, score-branch)` folded factor rows for token `t`. Every 128-vector entry has unit RMS, so all complete rows have
the same norm. Let `d_sa` be one of rung430's 512 unit-norm decoder atoms.

Compute a Frank–Wolfe approximation to

`min ||c-d_sa||_2^2 subject to c in conv(X_s(FIT)/||X_s(t)||)`

and separately to the signed symmetric hull `conv(X_s(FIT) union -X_s(FIT))`. Every Frank–Wolfe linear minimization
is global over the full frozen FIT token set, not a nearest-neighbor candidate pool. Retain the selected token IDs,
signs, nonnegative weights, weight sum, reconstruction check, objective path, and final dual gap. The stored decoder
is `c/||c||`. This normalization changes only the atom/coefficient scale gauge; the unnormalized `c` is the literal
convex-hull certificate.

Token roles are frozen from the rung430 split: FIT is token ID mod5 not equal to4; SELECT is mod5 equal to4. FIT-A is
residues0/1 and FIT-B residues2/3. Use 16 global Frank–Wolfe steps after the initial vertex. The deployed active count
remains k27 query plus k27 key, with the exact rung430 token supports, coefficients, and biases.

Arms:

- `U54`: unchanged rung430 CP54 artifact;
- `H54`: unsigned FIT-hull atoms;
- `SH54`: signed symmetric FIT-hull atoms;
- `RSH54`: the fixed relaxed point `c + .25*(d-c)` before normalization;
- `PH54`: signed hull of independently entry-permuted FIT rows, preserving each entry marginal while destroying
  whole-token alignment.

The same `SH54` projection is computed for rung430's frozen independent-restart seeds and separately from FIT-A and
FIT-B. These are stability diagnostics, not extra selectable arms.

## Observable atom profiles

Private Q/K factor coordinates have gauge freedom. Atom matching therefore uses frozen score profiles, not raw
decoder cosine. Select 16 FIT anchor tokens by seed43901 and offsets `(1,4,16,64)`. A query-atom profile is its exact
18-entry rotary score against every key anchor and offset; a key-atom profile reverses the roles. Profiles use the
real per-entry RMS normalization and are unit-normalized only after materialization. Match restarts by Hungarian
assignment on absolute profile cosine.

As an instrument check, apply independently seeded 2D phase rotations in every rotary plane, jointly to query and
key factors/atoms. These rotations commute with RoPE and preserve the exact score. Profile values and all reported
profile cosines must be invariant within `2e-5` maximum absolute error.

## Data and price

SELECT random token pairs/offsets provide the frozen score/product bridge. The 96-row FINAL role in
`mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json` has never been opened by rungs426/430 and is the binding
document consequence set. It measures full attention0-write error, six downstream consumers, and suffix CE.

Every arm has the same deployed bill as rung430 CP54: 15,583,320 raw bytes for two FP16 decoders and biases plus
50,257 query and key k27 FP16 coefficients and uint16 indices. Convex weights are training certificates, not runtime
inputs; the materialized decoder is charged. Native layer-0 Q/K maps must be unused by candidate generation.

## Frozen predictions

### A — valid isolated instrument

The parent bundle hash matches; U54 reproduces rung430 SELECT complete-pattern error within `.005` and SELECT CE
damage within `.002` nat; every hull certificate has nonnegative weights summing to one within `2e-6`, materialized
reconstruction error at most `2e-6`, monotonically nonincreasing objective, live global linear minimization, and
finite dual gap; all dtypes/shapes/bills hold; no-native-QK suffix replay relative squared error is at most `1e-12`;
FINAL is opened exactly once; and the legal rotary-gauge profile check is at most `2e-5`.

### B — the real signed hull is geometrically specific

For query and key separately, SH54 mean squared projection residual is at most `.90` times PH54's matched
entry-permuted residual. The real whole-token hull must help both sides; averaging them is not sufficient.

### C — anchoring preserves the actual computation

On FINAL, relative to U54:

- SH54 complete-pattern, full-write, and mean-six-consumer errors are each at most `1.25` times U54, and CE damage is
  at most U54 plus `.005` nat;
- RSH54 complete-pattern, full-write, and mean-six-consumer errors are each at most `1.10` times U54, and CE damage is
  at most U54 plus `.003` nat.

These are noninferiority bars, not claims that a constraint must improve fidelity at unchanged price.

### D — anchoring improves gauge-safe identification

For query and key separately, SH54's independently restarted median matched absolute profile cosine is at least
`.60` and at least `1.20` times U54's matched profile cosine. For the same primary atoms projected independently from
FIT-A and FIT-B, median same-atom profile cosine is at least `.80` on both sides.

## Strong null and routing

The strong null fires on A failure; if SH54 fails to beat PH54 residual on either side; if SH54 restart profile
stability is no better than U54 on either side; or if RSH54 FINAL CE exceeds U54 by `.020` nat or its full-write error
is at least `1.50` times U54.

- A/B/C/D true and null false: the convex-hull prior is both computation-compatible and identification-improving;
  license one fresh-corpus plus registered extraction/removal gate against U54 and the continuous quotient.
- A/B/C true, D false: the constraint is compatible but does not identify mechanisms; close convex-hull anchoring as
  the solution to atom nonuniqueness.
- B false: real token hull geometry has no matched-control specificity; close this Archetypal family at this object.
- C false: anchoring selects a convention that damages the computation; close it regardless of stability.

No threshold, hull step count, relaxation, token role, active count, atom count, or objective is tuned after outcome.
