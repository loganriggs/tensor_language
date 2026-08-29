# Hourly strategic review — 2026-08-29 04:35 UTC

## Outcome-changing update

The native-stream fallback branch is closed.  A rank-512 map fitted on native
length-one streams looked strong, but those streams came from the model being
replaced.  Two source-closure tests now fail with all controls intact:

| map input and fit | uncovered deficit, three roles (nat) |
|---|---:|
| embedding, covered-fit, deployable | `0.59560 / 0.67209 / 0.67172` |
| native length-one stream, covered-fit, oracle input | `0.17427 / 0.21358 / 0.21419` |
| compressed-program stream, native-fit | `1.08978 / 1.27276 / 1.26133` |
| compressed-program stream, self-refit for three iterations | `5.49867 / 5.61939 / 5.59476` |

The self-refit iterates were not converged—the relative changes were
`22.63 → 5.44 → 1.86`—but the registered weak question was only whether fitting on
the deployed input helped at all.  It made the deficit roughly five times worse.
The deployable fallback is therefore the rank-512 embedding map; the native-stream
result remains a diagnostic that native intermediate state contains useful
information, not an executable compression.

Family F completed all registered computation but terminated before result/receipt
publication.  The exact call census passed (2,400 optimizer steps, 9,600 backwards,
13,141 raw-logit returns and zero student native-MLP3 calls), but terminal validation
rejected a CUDA-computed polarization maximum when a CPU replay differed by more than
an absolute `2e-5`.  The authority, 99.3-MB program artifact and failure are preserved;
no result or receipt exists, so v1 earns no numerical credit.

Independent audit and a width-1152/K512 known answer localize this as backend-dependent
float32 reduction order, not a tensor mismatch: CPU and CUDA can differ by more than
`2e-5` in the *maximum absolute residual* while each independently satisfies the
registered `2e-5` relative identity gate by about two orders of magnitude.  V1 is
spent and will not be mutated.  Recovery requires a fresh hash-pinned v2 namespace.

Another agent launched the map-rank-512 price-frontier calculation after Family F
released the GPU.  This review does not duplicate that job.

## Honest fraction explained

These currencies cannot be averaged:

| Ledger | Current credit | What remains |
|---|---:|---:|
| Structural tensor inventory | 36/36 components | semantic and causal meaning |
| Strict whole-program storage-removal certificate | 5.3481% | 94.6519% uncertified |
| Strict named causal CE recovery | 10.923% | 89.077%, or 4.72714 nat, unnamed |
| Dense analytic interface substitutability | 99.8162% of its floor-relative denominator | compressed but not legible, sparse or edit-certified |
| Terminal extraction/removal/OOD actions | 0/68 | all 68 actions lack scientific outcomes |

The 99.8% interface result and the 10.9% named-circuit result answer different
questions.  The former says a dense learned interface can preserve behavior; the
latter says we still cannot name, extract or selectively edit most causal computation.

## Largest remaining gaps

1. **Composition remains the central failure.**  Individually useful component
   stand-ins interact badly when installed together.  In the current factorial
   localization, interactions consume 43–64% of cell effects, so independent local
   fits are not additive evidence.
2. **The early MLP group is the largest localized residual.**  MLP0–2 contribute
   0.728 of the 0.873 global ship nats by held-out Shapley allocation, and 1.078 of
   1.176 nats on novel-rare positions.  We do not yet have a reusable output port for
   that group.
3. **The deployable fallback is still expensive and inaccurate.**  Thirty-six
   independent rank-512 maps cost 42,467,328 floats and leave roughly 0.60–0.67 nat
   per uncovered position.  Their bases are also rotationally arbitrary.
4. **OOD transport is unestablished.**  The prose-derived rank-64 content basis
   captures only 16.6% of code variance, 32.2% of the code-local top-64 ceiling.
   More FineWeb skips establish sampling replication, not domain transport.
5. **No simplicity definition has yet earned selective-control credit.**  Storage,
   product count, rank, sparsity and local MSE are useful prices, but none has yet
   produced a held-out extraction/removal/OOD success in the 68-action ledger.

## Candidate actions and pruning

### A. Recover Family F under a fresh transaction

V1 is a preserved implementation failure, not a scientific result.  The narrow repair
is to require GPU and CPU polarization checks to pass the same relative tolerance
independently, without comparing their reduction-sensitive maxima.  A fresh v2 must
pin the v1 authority/program/failure hashes before loading them and independently
reconstruct every program tensor.  A reporting-only recovery is cheaper than
retraining but must label the missing v1 optimizer traces unavailable and nonpromotive;
a full v2 refit is cleaner but redundant if exact reconstruction succeeds.  Fit-only
KL still cannot open validation.

### B. Jointly factor the 36 deployable embedding maps

For site (j), fit

$$
W_j=A_jV_{g(j)}^T,
$$

where (g(j)) is either one global group, two fixed groups (attention and MLP), or one
group per site.  The exact optimum for each group uses the top eigenspace of

$$
M_g=\sum_{j:g(j)=g} C_j^T(G_j+\lambda I)^{-1}C_j.
$$

At rank 512, one global basis costs 21,823,488 floats (48.61% less than independent
factors); two bases cost 22,413,312 floats (47.22% less).  This is the broadest
currently available certified simplification opportunity.  It must be scored by
whole-program held-out CE, not just the joint regression objective.

### C. Construct a vector-valued downstream predictive state

Rows are controlled early-component interventions; columns are later residual
directions and fixed logit groups.  A low-dimensional state must predict sealed
prefix × suffix-reader cells and document-disjoint data.  This turns “rank” into an
operational number of causal variables needed for unseen compositions.

### D. Start one short terminal behavior circuit

Screen capitalization, number formatting and copy/continuation in late blocks, then
attempt one extraction/removal pair with matched negative examples, collateral CE and
OOD templates.  This is the fastest route to learning whether sparsity, product count
or shared-state rank actually helps an interpretation task.

### E. Test the admitted Family-F write as a causal port

Conditional on an uncalibrated Family-F program passing its registered fit gates,
compare native and candidate suffix responses to both signs and two amplitudes along
fixed directions.  If Family F fails, use its frozen score gradients only to test
whether a small stable context-conditioned gate dictionary exists; do not enlarge K
or tune the failed global support post hoc.

### Pruned or deferred

- More stream-input closure variants: two independent failures close the branch.
- Map rank above 512: rank 1024 costs more than dense storage and buys only
  `0.008` embedding-map nat or `0.024–0.037` oracle-stream nat.
- Independent local SAE/HOSVD/gauge experiments as the main line: they do not address
  the measured suffix interaction and composition failure.
- Scalar aggregate-CE Hankel crosses: the existing version failed and concealed
  sign/vector structure; only a prospective vector response panel is worth reopening.
- Duplicating the rank-512 table frontier: another agent already owns and queued it.

## Ranked top five

1. **Source-close a fresh Family-F v2 recovery.**  The expensive fit completed and
   its exact program exists, but v1 cannot publish.  A hash-pinned reconstruction plus
   fresh reporting is the cheapest lawful route to the causal-composition answer.
2. **Run the real 36-site shared-output RRR sweep.**  It offers nearly 2× storage
   compression across every site and a common coordinate system, with an exact global
   solution and a now-settled source-closed target.
3. **Build the vector downstream-response panel.**  It tests unseen compositions and
   can define reusable causal state, addressing the largest conceptual gap.
4. **Run one terminal extraction/removal circuit.**  It can move the currently empty
   practical-action ledger and validate which simplicity currency is useful.
5. **Run a Family-F finite-secant port test if admitted, otherwise a frozen gradient-
   stability diagnostic.**  This converts the current block-local experiment into an
   editability result or cheaply falsifies a context-router rescue.

## Highest-priority safe CPU action executed

While Family F owns the GPU, the simultaneous RRR implementation was extended to fit
one exact output basis per prospectively fixed group.  It now interpolates exactly
between one global dictionary, two attention/MLP dictionaries, and independent
site dictionaries while preserving site order and reporting literal prices.

Four new known-answer/adversarial tests verify:

- exact one-group replay of the global solver;
- exact one-group-per-site replay of independent reduced-rank regression;
- the 36-site one/two/36-basis prices;
- rejection of missing or unhashable group assignments.

The expanded suite passes `12/12`.  No model outcome was opened, and no GPU job was
duplicated.  After Family F failed terminal publication, the validator was narrowed
to the actual invariant: exact program tensors plus independent device-local relative
polarization gates.  A new regression test supplies deliberately different valid
backend maxima; the focused Family-F suite passes `58/58`.  Independent math red-team
agrees with the diagnosis and requires a fresh v2 namespace.  The next implementation
step is the hash-pinned reporting-only recovery protocol; the running rank-512 frontier
owns the GPU in parallel.
