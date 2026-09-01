# Hourly strategic review — 2026-09-01 20:36 UTC

## Current goal

Compile bilin18 into a smaller predictive, manipulable tensor program whose parts are justified by held-out
computation and causal downstream effects. For attention0 specifically, replace the architectural-head basis with
the smallest executable query/key/composition/payload program that preserves language-model behavior, then decide
whether its coordinates have stable semantics.

## What changed in the last hour

- Rung 424 found a near-lossless continuous six × six × 32 decomposition of the realized QK1 × QK2 × payload
  product, with +0.000200 nat SELECT damage, but retained every native generator.
- Rung 425 reproduced it on unused documents.
- Rung 426 found the first surviving cross-head sparse token vocabulary: G54 is 18.8446% smaller than 18 independent
  dictionaries, improves product error 1.4740→0.64375, and survives same-token permutation and no-native-QK tests.
  Its individual atom-identity predicate missed one raw-pair bar, while routed output and CE strongly depended on
  the learned coupling.
- Rung 427 reproduced the 426 document ordering on fresh rows; the G72-over-I72 CE margin remains positive but
  shrinks from .00317 to .00084 nat.

## Is the coupled sparse score experiment still the best next step?

Yes, with bounded claim level. It changes both the query/key factorization and the training objective, directly
answers the user's requested second section-14.3 extension, and resolves the specific ambiguity left by 426: whether
atom coupling is weak in raw pair geometry because factor MSE learned the wrong coordinates, or because no stable
discrete composition exists. It is cheap, uses the already validated physical execution path, and has controls that
separate real query×key relations from capacity.

It must not displace the continuous physical generator if it only improves interpretability while remaining around
+.02 nat damage. Rung 424 is roughly two orders of magnitude better in CE but has no saving; rung 430 is useful only
if it either moves sparse fidelity materially toward 424 or identifies stable atom-pair structure. The comparison is
therefore complementary rather than winner-take-all at this stage.

## Alternatives reconsidered

1. **Direct continuous composite generator now.** Highest adoption relevance because 424/425 are near-lossless.
   Retain as the next parallel scientific family after the short sparse screen. It must emit score modes without
   native Q/K and solve the full value broadcast, with matched ordinary-rank and existing shared-QK baselines.
2. **Downstream-62 metric first.** Could choose the right basis, but the universal damage ray makes raw certificate
   counts easy to game. Defer until there is an executable candidate; then use document-disjoint response vectors,
   tag permutation, and matched CE/price.
3. **Pure semantic inspection of rung-426 atoms.** Premature because D failed and rotations/restarts remain
   untested. Rung 428's atom-pair concentration and restart matching are the minimum gate before naming.
4. **More atoms/k/rank tuning.** Rejected for now: it would read SELECT and does not change the scientific object.
5. **Full Tucker/CP of the dense QK×payload tensor.** Retained only if sparse pair stability fails or as the direct
   continuous generator parameterization. Raw coefficient-space decompositions without downstream/product metrics
   duplicate old nulls.
6. **Head-6 characterization.** Mechanistically interesting after 423, but lower leverage for the global goal than
   deciding whether the new sparse vocabulary composes. It can proceed independently without blocking 428.

## Frozen decision after review

Run rung 430 once at its preregistered 512-atom k27+k27 and k36+k36 prices. Do not tune after SELECT. If coupled
training passes computation but not stability, retain the sparse generator and treat its atoms as a non-unique
coordinate system. If it misses computation or the strong null fires, close this sparse-composition budget and move
immediately to the direct continuous composite generator. If it passes both, fresh/OOD and 62-behavior tests precede
any semantic name or adoption claim.

This review changes no registered rung-430 bar, arm, seed, split, or objective. The number was repaired after the
review because the red-team lane had already claimed and executed rungs 428 and 429; no scientific content changed.

## 20:45 addendum — user correction: simplicity is validated by its consequences

The user correctly objected to wording that treated fewer parameters or bytes as the definition of simplicity.
The repository already has the stronger contract in `SIMPLICITY_CONSEQUENCE_VALIDATION_V1.md`, `FORMALISM.md`, and
the MLP0 dossier: **a simplicity measure earns trust only when its ordering predicts a named useful consequence on
untouched data at matched causal fidelity**. “Smaller” without that consequence is bookkeeping.

The project should therefore keep two objects separate:

1. `K(P)`, a vector of candidate resource/structure measures—standalone and amortized bits, operations, sequential
   depth, gauge-quotiented dimension, interface capacity, graph locality, conditioning, and certificate status;
2. `Y(P)`, the consequences the proposed kind of simplicity promises.

For the user's current goal, the primary consequence vector is:

- **OOD prediction/generalization:** a circuit description learned on one document/token population predicts
  activations, behavior, and intervention effects on unseen documents, shifted corpora, and held-out token classes;
- **circuit extraction:** a small declared set of variables/edges predicts a named behavior above difficulty- and
  shuffle-matched controls, including signed responses rather than only activation correlation;
- **selective circuit removal:** intervening on those variables reproduces the native target effect while changes to
  unrelated, non-descendant circuits remain below a preregistered collateral bound.

Composition and reuse are supporting requirements, not merely extra aesthetic goals. If two extracted circuits
cannot be installed or edited together with predictable effects, the representation has not separated their causal
dependencies. If one subprogram is reused across several circuits, its amortized price and intervention semantics
must be shared consistently rather than copied or redefined. Composition therefore tests whether extracted parts
are actual modules; reuse tests whether a proposed shared abstraction remains the same object across consumers.

Operationally, compare candidate measures only after matching validation causal distortion:

`C_j(P) < C_j(Q)` and `D_val(P) approximately D_val(Q)` must predict `Y_j(P) > Y_j(Q)` on untouched tests.

Different measures may win for different promises: bytes can predict storage, sparse typed graphs may predict edit
locality, causal-interface dimension may predict intervention transport, and prequential description length may
predict data efficiency. No one scalar is assumed to dominate. A candidate can stay on a storage frontier while
failing circuit usefulness, but it must be called storage compression—not a simpler interpretable program.

### Consequence for the live attention0 route

The direct continuous six/six/32 generator remains the next construction, but its acceptance ladder is amended at
the project level, not by changing any already-frozen rung:

1. independence and real execution: no native generator calls, complete producer/decoder price;
2. matched causal fidelity: fresh-document output, consumer responses, CE, and gauge invariance;
3. OOD transport: a second corpus or code distribution and held-out token/offset families;
4. extraction: predict at least one held-out named attention0 behavior from the proposed interface;
5. selective removal: remove that behavior with a native-matched signed target effect and bounded unrelated-circuit
   collateral;
6. composition/reuse: jointly install with the current frontier and predict interaction effects before measurement.

Thus a smaller physical continuous generator would be only a storage/execution result after steps 1–2. It becomes
evidence for the user's desired kind of simplicity only when the same representation makes steps 3–6 work better
than matched-fidelity ordinary rank, the sparse generator, and difficulty-matched controls. Future hourly reviews
will report these consequence ledgers explicitly rather than using “simpler” as shorthand for fewer bytes.

## 20:54 addendum — archetypal dictionaries and learning the simplicity prior

The user's Archetypal-SAE suggestion is a useful new sparse-family arm, but not an identifiability theorem for this
model. Fel et al. constrain decoder atoms to `D = W A`, where each row of `W` is nonnegative and sums to one, so
each atom is a convex mixture of observed activation rows. Their relaxed version uses `D = W C + Lambda` with a
bounded deviation from a centroid hull. This is principled when the intended latent concepts are extremal or
prototype-like points of the data geometry. It reduces the arbitrary freedom of placing decoder atoms anywhere.

It does not make arbitrary latents identifiable. The paper itself says the inductive bias must match the generating
process and evaluates “soft identifiability” on synthetic object mixtures. Its stability bound also contains a term
depending on the difference between the two learned convex weights, so convex membership alone does not force two
optimizers to select the same atoms. In our case there are three additional mismatches:

- rung 430 uses signed sparse codes, whereas the archetypal conic interpretation assumes nonnegative codes;
- useful Q/K roles may be signed contrasts between token groups rather than extreme token rows;
- convex-hull membership in native Q/K coordinates is not invariant to a function-preserving Q/K gauge.

Therefore the closest replication arm is worth running, but the more project-specific version should constrain
atoms in **induced score-profile space**: a row is what one query token scores against a frozen bank of key tokens
and offsets (and analogously for keys). Convex mixtures there are mixtures of observable attention roles and are
invariant to private Q/K coordinate rotations. A downstream-response-profile hull is an even stronger later version.
Strict, relaxed, symmetric signed-hull, and ordinary rung-430 arms should be matched for active count and complete
price. Success requires not merely higher restart cosine but better held-out score/product/CE, atom-pair stability,
shifted-token/document transport, extraction, and selective removal collateral.

The user's second idea is the general form of this move: **learn which proposed simplicity measures predict useful
consequences, then optimize programs under those measures**. This is legitimate bootstrapping rather than cheating
if it is nested and prospective:

1. On meta-training circuit families, fit small consequence-specific predictors from the simplicity vector `K(P)`
   to OOD transport, extraction, edit specificity/collateral, and composition/reuse outcomes.
2. Select measures and weights only on different validation circuit families, not different examples of the same
   circuit. Prefer monotone pairwise order predictions at matched causal distortion to a free scalar reward.
3. Use the selected measure to search a new program on training data.
4. Freeze the program, measure, extractor, edit, and semantic labels before opening held-out circuit families,
   shifted data, intervention types, and compositions.
5. Include adversarial candidates: equally small random rotations, label/tag permutations, causally wrong
   reconstruction matches, duplicated “reusable” modules, and programs that hide native calls or collateral.

The held-out result is not “the optimized objective became small.” It is whether the simplicity ordering predicted
the consequence vector before measurement. Keep separate predictors unless evidence supports scalarization. The
learned simplicity rule itself has a description length and must beat bytes, rank, sparse graph locality, causal
interface dimension, and shuffled-rule controls. This protocol is now a standing item in hourly reviews: ask not
only for alternative program decompositions, but whether the accumulated circuit ledger can train and prospectively
falsify a better simplicity prior without reusing held-out circuit or data roles.
