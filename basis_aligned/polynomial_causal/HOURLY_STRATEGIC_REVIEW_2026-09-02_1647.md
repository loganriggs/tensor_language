# Hourly strategic review — 2026-09-02 16:47 UTC

## Circuit targets and full goal

The goal is a smaller executable tensor program that predicts fresh and shifted text, composes when several
replacements are installed, supports selective removal/swapping/editing, and is cheaper under literal storage,
compute, edge, state, and program prices. A useful circuit decomposition must eventually establish:

1. what information is read, what operation is performed, what is written, and what later computation uses it;
2. grouping across attention heads or MLPs when later computation treats their parts as the same variable, and
   splitting a native module when its parts serve different tasks or compositions;
3. held-out and OOD prediction of activations and signed causal effects;
4. extraction or sufficiency: an executable circuit, or an explicit interface plus necessary background;
5. selective manipulation that changes the intended behavior while preserving unrelated behavior and accounting
   for redundancy and interactions;
6. predictable composition and reuse of shared and task-specific pieces; and
7. stable identification across documents, corpora, plausible gauges, and refits, or by operational equivalence
   under downstream readers.

Rank, quantization, reconstruction error, and average CE may price or control an already identified object. They do
not by themselves discover, group, split, name, extract, or selectively manipulate a circuit.

## What changed since 14:33

- Rung491 identified attention1 as the only named residual source necessary for both the token-only and interaction
  MLP1 branch responses on held-out intervention outcomes. It was a source attribution, not a standalone circuit:
  attention1 alone was insufficient.
- Rung492 then falsified the stronger input-edit interpretation. Output-level bilinear attribution did not survive
  the corresponding physical change to the normalized MLP1 input. This closed the tempting but invalid leap from an
  exact output algebra to an editable upstream mechanism.
- Rung493 tested the apparent T/I grouping physically. Attention1 merging removed only about 26--28% of the T/I
  contrast, whereas merging at MLP1 removed 84--95% for every branch pair. MLP1 is a generic downstream bottleneck,
  not a T/I-specific circuit boundary. The previous 51--62--79% geometric pattern is now descriptive only.
- Rung494 tested whether seven three-MLP subset effects compose through one occurrence-specific monotone scalar
  readout. It failed at half strength, where ordinary addition was already very accurate. It worked descriptively at
  1.5x strength, but part of that win used endpoint clipping and one document half failed stability. The separate
  quadratic-response falsifier also failed at every scale. These results map a nonlinear saturation regime but do
  not identify a reusable circuit.
- Rung495 changes the object: it splits each attention1 head into the seven exact nonempty interaction terms among
  score branch 1, score branch 2, and value/output, giving 63 raw write pieces. It groups those pieces by their
  62-circuit downstream-use derivatives rather than by head, rank, or reconstruction.
- Implementation review caught an important mathematical error before outcomes opened. The raw attention write is
  normalized together with the incoming residual before MLP1, so raw pieces cannot be independently pushed through
  MLP1 and then added. The corrected rung differentiates the real normalized suffix with respect to the complete raw
  attention1 write. The 63 raw pieces and their first-order effects both close exactly by linearity; any claimed
  finite interchange remains a separate required experiment.

## Is rung495 still the highest-information route?

Yes. The last four causal tests repeatedly show that native heads, whole MLP branches, output-space algebra, and
low-dimensional response fits are not reliable semantic units. Rung495 directly tests the user's proposed remedy:
look below heads, then merge or split pieces according to what later circuits can distinguish. It can change the
cross-boundary grouping, within-head splitting, held-out-prediction, and stable-identification targets. A pass only
identifies a candidate operational quotient; it does not satisfy extraction or selective manipulation until a
finite natural-state interchange succeeds.

The corrected run is currently live through the managed runner. It selected on documents 0:250, confirms without
reselection on 250:500, uses one-sided circuit-label controls, conditionally checks shifted positions, and opens
documents 500:1000 plus 30 validation circuits only after all discovery conditions pass.

## Confound audit

- **Normalization:** fixed before observation by differentiating through the actual RMS normalization and suffix.
- **First-order versus finite changes:** a gradient match is explicitly only a screen; it cannot license a swap.
- **Generic common mode:** signatures must survive removing each MLP0 branch's mean fingerprint.
- **Circuit difficulty:** sixteen one-sided circuit-label permutations test whether apparent matches are generic
  high-response circuits rather than shared use.
- **Position and token difficulty:** a preliminary pair triggers sixteen fixed-pair position shifts; failed controls
  cannot be dropped.
- **Post-selection:** the pair is selected on the first half and frozen for confirmation and conditional validation.
- **Gauge:** the raw QK and OV coordinates remain non-identifiable, but the complete finite factor arms, raw residual
  writes, and downstream derivative functionals are invariant to compatible internal coordinate changes.
- **Loss nonlinearity:** CE is differentiated only as a local downstream-use measurement; no sum of CE ablations is
  treated as a finite causal prediction.
- **Precision and instrument liveness:** exact write/Moebius/gradient closure, call counts, data fingerprints, and
  nonzero controls are registered; CPU tests and preflight passed.
- **Evidence scope:** conditional documents 500:1000 are held out from rung495's selection, but are not a new corpus.

## Genuinely different next moves, ranked

1. **Finite interchange for a passing cross-head pair.** Exchange the frozen raw pieces inside the complete
   attention1 write, recompute normalization and the suffix, and test signed target effects plus preservation of
   unrelated circuits. This advances extraction, selective manipulation, reuse, and composition. It dies if the
   gradient equivalence does not survive finite swaps or if unrelated circuits move comparably.
2. **If no pair passes, split each QK score branch into its query-side and key-side token functions and group those
   halves by downstream use.** Earlier weight-space shared-half tests were stable as estimators but did not establish
   semantic reuse. Conditioning the query/key halves on the 62 downstream circuits changes the object rather than
   tuning their rank. This advances finer cross-head grouping and computational specification. It dies if matches
   are no more stable than circuit/position controls or if recomposed scores do not transport.
3. **Predictive-state causal quotient across module boundaries.** Treat two pieces as the same state only if every
   registered downstream intervention distinguishes them by less than a frozen tolerance, then minimize the state
   partition and test held-out transitions. This advances operational identification and a reusable interface. It
   dies if the quotient is unstable under held-out circuits/documents or fails finite state substitutions.
4. **Signed composition algebra for the MLP0-to-MLP1 path.** Use exact factorial effects to ask whether a small named
   algebra predicts joint T/C/I/S edits across contexts, rather than fitting a scalar curve to one query circuit.
   This advances composition and within-MLP splitting. It dies if held-out joint edits require occurrence-specific
   lookup or if selective removals disturb unrelated circuits.
5. **Lower-bound/falsifying toy models.** Construct two networks with identical local reconstruction, rank, and
   gradient profiles but different finite downstream interchange behavior. This would prove which observations can
   never identify the desired quotient and prevent more proxy drift. It advances stable identification by ruling
   out insufficient criteria, but is secondary while rung495 directly tests the live object.

The attention-piece route remains first because it asks the central grouping/splitting question directly and has a
registered causal successor. No rank or precision sweep is licensed by any current result.

## Live continuation

Rung495 is running with sustained GPU utilization. At landing, score it exactly as registered. A pass requires a
board claim and preregistration of the finite interchange before yielding; a null requires a board claim and an
actually begun query/key-side downstream-use split. A queue drain, completed receipt, or explanation is not a pause
condition while the program-level goal remains unfinished.
