# Real 36-site shared-output RRR v1

**Status:** prospective design; no authority, result, failure, or receipt namespace may
be opened until the runner and tests are committed, pushed, and independently audited.

## Question

The deployable fallback at each of the 36 attention/MLP sites is a linear map from the
current token embedding to that site's 1-token output row.  Independent rank-512 maps
cost 42,467,328 float32 values.  This experiment asks whether those maps use a common
output subspace, so one output dictionary can be stored once rather than 36 times,
without losing the held-out CE gain that made rank 512 useful in the high-table regime.

For site \(j\), covered-token embedding matrix \(X\), and native 1-token output table
\(Y_j\), the registered grammar is

$$
Y_j \approx X A_j V_{g(j)}^\top,
\qquad V_g^\top V_g=I.
$$

The global arm has one \(V\).  The typed arm has one basis for all attention sites and
one for all MLP sites.  The independent arm has one basis per site.

## Frozen data and split

- Fit/coverage rows: `.rowcache/fineweb_n96_skip80.pt`, SHA256
  `94bc1fb3e3a6a061541e555295e0af8c50ae6068fdff84e95a69c25844091eda`.
- Evaluation roles, never used to choose bases, ranks, ridge, or allocations:
  - `fineweb_n192_skip7000.pt`, SHA256
    `d66c1ee7807bc6b9bd7d0ddba5cdd7e3bc64926b00320a10675a2f817d67128c`;
  - `fineweb_n192_skip11000.pt`, SHA256
    `b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868`;
  - `fineweb_n96_skip1200.pt`, SHA256
    `21707551f35d13818c10ac59e12e9445ef076d0522371fe779691bfab719d34f`.
- Rows are truncated to 257 tokens.  CE is scored at target positions 64--255.
- The fit rows define exactly 5,419 covered token types.  Covered tokens retain exact
  native 1-token site rows.  The fitted map is applied only to uncovered types.
- All roles are discovery roles.  There is no validation/final promotion in v1.

## Exact fit objective

Use float64 sufficient statistics

$$
G=X^\top X,qquad C_j=X^\top Y_j,
$$

and the scale-relative ridge

$$
\lambda=0.01\frac{n}{1152},
$$

where \(n=5419\).  This is the previously deployed ridge convention and is invariant
to exact row replication when \(G\), \(C_j\), and \(n\) scale together.  Residual
coordinates remain the model's physical coordinates; no nonorthogonal input gauge is
claimed.

For any prospectively fixed group of sites, the exact shared output projector is the
top-\(q\) eigenspace of

$$
M_g=\sum_{j:g(j)=g} C_j^\top(G+\lambda I)^{-1}C_j.
$$

Ranks are \(q\in\{64,128,256,512\}\).  Eigenvectors are only a gauge representative:
signs and rotations inside tied eigenspaces have no semantic meaning.  Compression and
prediction claims attach to projectors and deployed coefficient maps, not columns.

## Frozen comparators

For every \(q\), report:

1. independent site bases at the same rank \(q\) (same per-site dense multiply count,
   larger storage);
2. one global shared output basis at rank \(q\);
3. separate attention and MLP output bases at rank \(q\);
4. the strongest independent-site allocation at exactly the global arm's storage;
5. the strongest independent-site allocation at exactly the typed arm's storage.

The matched-storage independent comparator is not a rounded common rank.  If the shared
grammar has \(g\) output dictionaries, its map storage is

$$
S_g=q(36\cdot1152+g\cdot1152).
$$

Each independent rank-one term costs \(2\cdot1152\) floats, so the exact number of
rank slots is \(R=q(36+g)/2\).  Allocate the \(R\) slots by the globally largest
fit-only marginal eigenvalues of the 36 independent merit matrices, with deterministic
site then eigen-index tie breaking.  The selected ranks must be spectral prefixes and
their literal float count must equal \(S_g\) exactly.  Validation CE never chooses the
allocation.

The same-rank global/typed comparison has equal per-site factor multiplies, but the
typed arm stores 2.70% more map floats.  One additional direct storage-matched pair is
therefore frozen: global rank 494 versus typed rank 481, since
`37 * 494 == 38 * 481`.  This pair isolates whether architectural typing helps when
both shared grammars receive exactly the same map storage.

Two nonpromotive legacy coefficient-SVD anchors at ranks 64 and 512 reproduce the
settled runner's numerical object.  They validate wiring only; the scientific
independent comparator is optimal predictive RRR, which is a different factorization.

## Measurements

For each arm, rank, and role report:

- covered, uncovered, and all-position CE with exact token counts;
- all-position CE relative to same-rank independent and exact-price independent;
- fit-only explained penalized merit and residual fraction;
- map float count, full-program float count including the fixed full tables, bytes,
  and deployed dense multiply count;
- rank vector for an exact-price independent arm;
- projector orthogonality/idempotence, group eigengaps at the retained boundary, and
  finite-value checks;
- exact physical model-call ledger.

Compiled evaluation uses the explicit model facade with all 36 attention/MLP writes
supplied by the program.  A write is the covered table lookup or the uncovered
factorized embedding map.  It never invokes a native attention/MLP.  The attention
`v1` bus is returned as a correctly shaped zero sentinel and ignored by every compiled
attention dispatcher; a known-answer test must show this autonomous path matches the
legacy all-site post-hook replacement, whose native `v1` is observationally inert once
all 18 attention writes are replaced.  Embedding, residual scalar mixing, RMSNorm,
and unembedding/softcap remain explicit native tensor primitives and are not counted
as removed parameters by this experiment.

The fixed table price is charged to all arms.  The primary causal currency is held-out
whole-program CE.  Local fit merit is diagnostic only.

## Registered decisions

- **E2.1 storage/prediction pass:** on all three roles, at least one global shared arm
  must be no more than 0.01 nat worse than same-rank independent and must beat its
  exact-storage independent comparator by at least 0.01 nat.  Otherwise a common
  output dictionary has not earned its storage advantage.
- **E2.2 architectural split pass:** the typed-rank481 arm must improve all-position
  CE over global-rank494 by at least 0.01 nat on all three roles at exactly equal map
  storage.  Same-rank typed/global deltas are reported separately as equal-compute,
  2.70%-different-storage diagnostics.  Otherwise attention/MLP typing has not earned
  a second dictionary.
- **Controls:** coverage is exactly 5,419; covered CE is identical across all map arms
  within `1e-6`; same-rank independent q64 and q512 reproduce the corresponding
  published full-table frontier rows within 0.002 after that frontier publishes; all
  exact-price arms match their target float budget exactly.

A pass is a useful compression result, not semantic interpretation.  E2.3 remains a
separate prospective sparse-gauge stability and coordinate-intervention experiment.

## Lifecycle and resource contract

- Fresh exclusive namespace:
  `shared_output_rrr_real_v1_{authority,results,failure,receipt}.json`.
- Authority is published before any row tensor or model load.  It pins the committed
  source tree, this preregistration, runner/tests, core/tests, exact row bytes, model
  identity, ranks, ridge, groups, arms, call schedule, and resource ceilings.
- Result publication is create-only.  A failure spends v1 and produces no receipt.
  The receipt is the final filesystem write and exactly joins authority and result.
- Ceiling: one GPU, 75 minutes wall time, 16 GiB peak allocated CUDA memory, no
  optimizer/backward calls, and the exact registered native/logit forward census.
- No result may be interpreted before semantic replay of schemas, hashes, prices,
  token counts, call counts, controls, and gates.

These repeatedly exposed evaluation roles support an eight-hour discovery decision,
not a new generalization claim.  Any candidate promoted to E2.3 must be frozen and
rerun on a new registry-excluding whole-document role, with a second fresh role if the
first is used to select coordinates.
