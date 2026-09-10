# Do the task-readable quadratic functions survive an actual removal test?

The attention-cross program is a valid null: it meets the original P control
limit but carries only about8%/20% of the full temporal/iswas signed effects.
Its other components remain descriptive, with no fallback promotion. The earlier
dual-reader interchange retained own effects but failed its P no-worse bar.
This next test addresses a different requirement: removal of a fixed computation,
rather than transferring a donor-minus-base response.

Freeze the same task readers C=[C_A;C_B] and minimum-norm dual D with CD=I.
For MLP1, write y=b+W phi(n), phi(n)=(Left n)*(Right n). Define the two actual
quadratic output programs g_t(n)=D_t C_t W phi(n). Their zero is fixed by the
homogeneous bilinear numerator, phi(0)=0; native output bias stays in background.
No data-dependent centering, neutral reference fit, or posthoc offset is allowed.

Removing g_t is a STATIC native weight change:
W_without_t = W - D_t(C_t W), with b unchanged.
Removing both uses W-D(CW). The two edits commute in exact arithmetic, preserve
the opposite local reader, and have an explicit original-weight correspondence.
Use a separately edited in-memory copy or temporary scoped replacement, restore
original weights after every arm, and never change the stored checkpoint.

Why another test is necessary: interchange uses differences g(x')-g(x). Reassigning
a constant between a component and background leaves those differences and
interchange behavior unchanged while changing removal. Interchange alone does
not validate the natural zero or necessity of a proposed circuit. The fixed
homogeneous weight definition above supplies a falsifiable choice, not a claim
that its zero is already semantically identified.

Data: same saved48 target pairs and16 temporal P pairs, both base and donor
endpoints. Native, remove-A, remove-B, remove-AB =16 forwards and512 sequence
evaluations. No new rows, fit, rank, dose, site or bias-offset selection. All
native errors retained. This is opened text, not pristine OOD.

A: bound source/reader/checkpoint hashes, exact counts, no biases changed, original
weight restoration; FP64 weight-edit versus output-subtraction identity abs<=1e-8
and relative<=1e-9; single/joint edit composition and opposite-reader preservation
abs<=1e-8; deployed native MLP output versus corresponding output-subtraction
oracle abs<=1e-3 and relative<=1e-5 on all valid prefix positions. Compare the
same unchanged MLP1 inputs across arms. Native final margins and P baselines
replay the parent within abs1e-3/relative1e-5. Keep any invalid receipt unchanged.

B: each single removal suppresses its OWN task's native donor-minus-base margin
contrast: absolute signed retention<=.10 AND contrast RMS/native contrast RMS
<=.25. On the OTHER task, contrast-vector relative error<=.10 and its absolute
endpoint top1 predictions remain unchanged. This tests loss of cue sensitivity,
not merely lowering one token's logit or reversing the decision.

C: joint removal meets the same suppression bars on BOTH tasks. On temporal P
controls, every single/joint arm has mean full-vocabulary teacher KL<=.001,
p99<=.01 nats/token, and zero native top1 changes across BOTH endpoints. These
are prospective absolute removal bars, not a redefinition of the failed
interchange no-worse comparison. Iswas-specific control families remain missing.

Report all task contrast retention/errors, endpoint KL/top1 changes, and P
collateral even on failure. No repair by adding a fitted constant, changing bias,
choosing a weaker dose or expanding to another site. A failure rejects promotion
of this particular task-reader quadratic split as independently removable
circuits; it does not prove all circuits impossible. A pass still needs fresh
OOD, an independently executable shared producer and better structural cost.

Managed GPU only, alarm600s. No training, gradients or persistent weight updates.
Charge all545902902 native parameters, readers/writers, temporary edited matrices
and retained prefix/suffix. Equal parameter count is not structural reduction.
Reuse the saved row manifest, existing backend/captures/scorer and small weight
intervention primitive; do not build another forward executor.
