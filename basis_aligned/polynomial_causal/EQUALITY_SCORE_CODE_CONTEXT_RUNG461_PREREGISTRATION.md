# Rung 461: context decomposition of the code score transplant

Status: prospective explanatory diagnostic, frozen after rung 460 and before running any context-cell score
transplants. The `ood_code` role has already been opened by rung 460, so this experiment cannot confirm OOD
generalization or repair rung 460's failed response-margin clause.

## Question

Rung 460 transplanted L5H5's equality score into L8H4's payload and recovered 91.92% of L8H4's positive code CE
effect, with strong interchange and score-geometry controls. Its response cosine was high on positive positions but
also high on matched-negative and off-task positions, so the registered task-specific cosine margin failed.

The frozen explanatory hypothesis is that MLP9 uses a broadly shared response direction whose **strength** is gated
by equality context. Natural rung 457 predicts two context orderings: equality terms matter more at far than near
matches, and more when there is exactly one previous matching token than when there are several. Rung 461 asks
whether the selected score transplant follows those orderings on code.

## Frozen object, data, and arms

- data: the same 192-document `ood_code` role, with reporting halves `0:96` and `96:192`;
- source score: L5H5;
- target payload: L8H4;
- reader: MLP9;
- scale: rung 459's natural-fit `L5H5->L8H4.score_ratio`, unchanged;
- analytical arms: base (remove both terms), reference (restore native L8H4), and score hybrid (L5H5 score times
  the frozen natural scale and L8H4 payload);
- context cells: `near_positive`, `far_positive`, `one_predecessor_positive`, and
  `multiple_predecessor_positive`, using the already-hash-bound masks built by the rung-457 mask function;
- controls: native-versus-empty analytical replay, exact factor reconstruction, all-positive and off-target
  companion reports, exact call census, and SEALED-closure report.

There is no pair, reader, factor, scale, mask, threshold, or subgroup search. Do not run the L7H3 control again,
payload/whole hybrids, branch swaps, other downstream readers, another corpus, or SEALED attention0 outcomes.

## Computations

For a context cell `C`, let `N_C` be its predicted-token count and define:

`native_stake(C) = [sum CE_base(C) - sum CE_reference(C)] / N_C`

`hybrid_effect(C) = [sum CE_base(C) - sum CE_hybrid(C)] / N_C`

`recovery(C) = hybrid_effect(C) / native_stake(C)` when the native stake is positive.

Here CE is the usual per-token cross-entropy loss. A positive native stake means restoring L8H4 improves the model
relative to removing both equality terms. A recovery of one means the transplanted early score restores the same
average improvement.

For the MLP9 response, let `r = MLP9_reference - MLP9_base` and
`h = MLP9_hybrid - MLP9_base`. Report:

- direction similarity `sum(r*h) / sqrt(sum(r^2) sum(h^2))`;
- reference-relative error `sqrt(sum((h-r)^2) / sum(r^2))`;
- raw response size `sqrt(sum(r^2) / (1152 N_C))` and the analogous hybrid size.

The raw response size is the root-mean-square change in one MLP9 output coordinate. It is not normalized by MLP9's
total write, so sizes can be compared across context cells without hiding amplitude differences in a moving
denominator.

## Registered predictions

### A. Instrument

All source/model/row/role/mask hashes and frozen identities hold. Empty replay relative squared logit error is at
most `1e-12`, exact factor reconstruction relative squared error is at most `1e-10`, and the exact expected number
of analytical forwards is reported. SEALED attention0 outcomes remain unopened.

### B. Natural context ordering transfers to code

For both the native stake and hybrid effect, pooled `far_positive > near_positive` and
`one_predecessor_positive > multiple_predecessor_positive`. Each of the four inequalities also has the same sign in
both fixed document halves. This is the primary prospective test.

### C. The hybrid tracks context-dependent causal strength

Across the four pooled context cells, Spearman correlation between native stake and hybrid effect is at least `.80`.
The correlation is positive in both halves. Every pooled cell has positive native stake and positive hybrid effect;
every pooled recovery lies between `.20` and `1.50`.

### D. A shared direction survives the context split

Every pooled context cell has MLP9 response cosine at least `.65`, reference-relative error at most `.75`, and raw
reference and hybrid response size at least `1e-4`. Each half has positive cosine in every cell. This tests a broad
response direction; it is not a task-selectivity claim.

### E. Strength carries information that direction alone missed

Across the four pooled cells, the largest native stake minus the smallest is at least `.01 nat`, and the largest
raw reference response size divided by the smallest is at least `1.10`. The hybrid preserves the sign of both
registered context contrasts in raw response size. This supports—not proves—the amplitude-gating explanation.

The strong null is instrument failure; any pooled context cell with native stake at most zero or hybrid effect at
most zero; Spearman correlation between pooled native stakes and hybrid effects at most zero; or no registered
context contrast having the natural-text sign for either native stake or hybrid effect.

## Claim boundary and successor

Even a full pass says only that the already-selected code transplant has an interpretable context-dependent
strength law on already-open data. It saves zero parameters and does not establish independent OOD confirmation.

If B--E pass, freeze the context/amplitude law and find a genuinely unused corpus or behavior role for a new
prospective confirmation before splitting the QK branches. If the primary ordering fails but C/D pass, retain a
broad code matcher with an unresolved gating law and test a different, preregistered context variable. If C or the
strong null fails, treat rung 460's code causal recovery as an aggregate-only result and do not promote the shared
matcher beyond natural text.
