# Does a symmetric bilinear rewrite preserve independent operand interventions?

Fix MLP1 and the existing 48 temporal/iswas target pairs plus 16 temporal P pairs.
No new fit, rank, site, dose, text selection or offset. All text is already opened.
Prior MLP4 response factorial and the saved MLP11 symmetric-tensor audit are known.
This tests a different claim: the legal intervention domain of a function rewrite.

For normalized inputs n_b,n_d, define
L=Down[(Left n_d)*(Right n_b)]+bias and
R=Down[(Left n_b)*(Right n_d)]+bias.
Native branch interchanges are independent Left/Right projection-output hooks.
The symmetric extension is S=(L+R)/2. In an output reader, its missing contribution
is n_d^T skew(Left^T diag(C Down) Right) n_b. That contribution vanishes on tied
inputs but need not vanish under independent operand interventions. Equivalence
of diagonal execution does not license silently identifying these interventions.
RMS inputs stay as captured native normalized states; downstream norms are live.

Each population uses two native captures and six base-context forwards: zero,
Left-only, Right-only, both native branches, direct donor MLP output, symmetric
mixed output. Total16 model forwards/512 sequence evaluations. All positions
through each semantic endpoint are edited; padding is preserved. Native parameters
and input/background dependencies remain charged. This is an intervention-fidelity
screen, not a new claim that one operand has a named semantic role.

A (instrument): source/row hashes, counts, finite values, unchanged MLP inputs,
restored hooks, zero/native and both/direct-donor final full-logit agreement
maxabs<=1e-3 AND relative Frobenius<=1e-5. Independently evaluated native branch
outputs versus FP64 product formula maxabs<=1e-3 AND relative<=1e-5; FP64
symmetric/skew reconstruction maxabs<=1e-8 AND relative<=1e-9. Native baseline
paired margins replay the prior within1e-3/1e-5. Eight bounded CPU controls include
a live skew-only counterexample and restoration on normal/exception exits.

B (distribution prediction): for EACH Left/Right native reference, on temporal,
iswas and P panels separately, the symmetric extension has mean endpoint teacher
KL<=.001, p99<=.01 nats, and zero top1 differences. This compares the candidate
with the actual edited model, not with an unedited unrelated-behavior baseline.

C (causal prediction): for the same six comparisons, the full-vocabulary centered
logit change S−base predicts the native branch change within .01 relative RMS.
If reference norm<=1e-8, require absolute error norm<=1e-8 instead. Center each
logit vector across vocabulary before taking differences; no fitted decoder.

If B or C fails, preserve the null: this symmetric off-diagonal extension cannot
represent the declared native operand-edit API at the registered fidelity. It does
NOT reject symmetric rewrites restricted to tied-input edits, nor prove another
explicit intervention translation impossible. Report the complete local skew
size and native left/right/both task-margin effects descriptively, no alternative
arm promotion. Symmetric numerator rank is not a causal equivalence certificate.
Even a pass establishes no fresh OOD identification or structural simplification.

Managed GPU only, alarm600s, no model training or persistent updates. Use existing
backend, saved row manifest and scoring machinery; no new forward executor.
