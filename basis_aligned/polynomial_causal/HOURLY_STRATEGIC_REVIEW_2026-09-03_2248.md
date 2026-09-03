# Hourly strategic review — 2026-09-03 22:48 UTC

## Goal and success criteria

The goal is a smaller transparent tensor program for bilin18 whose pieces correspond to computations the model uses.
A circuit should eventually provide: (1) a precise behavioral computation, (2) grouping across heads or MLPs and
splitting within them when the behavior demands it, (3) held-out and out-of-distribution prediction, (4) executable
extraction or sufficiency, (5) selective removal or editing, (6) reuse and composition, and (7) stability across data
and harmless changes of internal basis. Rank, reconstruction error, parameter count, storage, and cross-entropy cost
are prices or diagnostics; by themselves they do not establish any circuit goal.

## What changed since 21:48

- R585's second repair was blocked because authorized row IDs did not guarantee correct tokens, directions, inserted
  vectors, or primitive logits. A third repair bound all of those quantities to the frozen computation.
- The exact-byte review of that third repair passed 85 producer-side tests and 11 managed-launcher tests, then found
  one new blocker: a completed result could attach arbitrary text to an `invalid_instrument` terminal. The review and
  three executable attacks are frozen in commit `30f720470`; no model outcome was opened.
- Iteration 4 now reconstructs FIT and SELECT instrument-failure lists from the retained endpoint, factor, intervention,
  and structural evidence. It uses canonical ordering so the same measurements produce one exact list.
- Some implementation checks use full intermediate tensors that are not retained. Native full-attention
  reconstruction, replay-versus-native full-logit equality, incomplete factor capture, live-factor reconstruction,
  hook-write correctness, and nonfinite values now abort before a result can be published. They cannot become editable
  explanations for a scientific null.
- Structural controls now test equality of the frozen inserted score-by-value tensors. This is the computation actually
  saved in the evidence package and can be recomputed from the saved factors; it no longer relies on discarded
  downstream logit vectors.
- The builder's stable candidate passes 52 owner tests and 11 adapter tests. Parent reproduced those tests, all 16 live
  versions of the first blocker contracts, both gates and preflights, and both zero-model dry runs. Two additional
  completed-package tests for the unreconstructible attention and replay checks are being added before hashes freeze.
- R584 remains a separately repaired numbered-list downstream-MLP experiment with 48 model-free tests, awaiting a
  new independent review. Its earlier FIT result was a genuine coarse-basis null: all twelve pieces were active, but no
  piece met the full selective-successor gate.
- Other agents found compact late-MLP programs and shared input directions. Those results are useful descriptions of
  implementation cost and composition error, but they have not yet attached the directions to behavior-level circuits
  or shown selective removal. They therefore do not replace R585 or the next behavior-specific wave.

## Is R585 still the highest-information action?

Yes. For recipient prompt $x$, donor prompt $y$, site $h$, and registered source role $r$, it crosses a continuous
equality-gated attention score $e$ with a projected source value $u$:

$$
t_h^{yx}=\sum_r e_h^y(r)u_h^x(r),\qquad
t_h^{xy}=\sum_r e_h^x(r)u_h^y(r),\qquad
t_h^{yy}=\sum_r e_h^y(r)u_h^y(r).
$$

The first two interventions separately change source selection and copied content; the third changes both. Selector
edits and payload edits make opposing predictions, and active answer-preserving controls test whether the intervention
is merely a broad contextual write. This directly tests whether pieces inside several attention heads implement one
reusable selector-by-payload computation. It is not a rank-reduction experiment.

## Confounds rechecked

- **Counterfactual meaning:** token identities and semantic source, payload, and query positions are joined to the
  frozen authority rather than trusted from labels.
- **Multiple mediators:** score-only, value-only, and joint effects are observed directly. A joint effect is not inferred
  by adding two separate patching effects.
- **Changed later states:** all recipient and donor factors are captured before interventions; both layer-8 heads use
  the same unedited layer input.
- **Evidence-derived decisions:** every publishable terminal and clause is reconstructed from exact phase evidence.
  Unretained implementation checks abort instead of producing an unauditable null.
- **Active controls:** control effects count only when the inserted residual-stream tensor is nonzero. Logit-margin and
  vocabulary changes are normalized by like-for-like FIT target scales, not by intervention norm.
- **Native head basis:** even a held four-site result would identify this score/value factor only within the tested
  sites. It would not prove that a whole head is an atom or that the factor is unique.

## Alternative mathematical and experimental routes

1. **Downstream-equivalence quotient:** merge factors from different heads when no registered downstream continuation
   distinguishes their interventions; split a head whenever continuations do distinguish its pieces.
2. **Block-term tensor recovery:** construct the donor-score $\times$ recipient-value $\times$ downstream-readout
   tensor from valid R585 factors, test uniqueness conditions, then subject recovered cross-head blocks to the same
   causal transfer and removal tests.
3. **Pending-opener downstream-use split:** separate pending-delimiter state from generic punctuation/context using two
   independent counterfactual constructions and active unrelated controls.
4. **Numbered-list successor split:** separate visible-label identity, successor action, and copy/arithmetic-conflict
   use rather than splitting only by MLP or by self/cross algebraic terms.
5. **Weight-space DAS:** after meaningful circuit datasets exist, search for the smallest rotated bilinear input/output
   subspaces that support valid interchange, then translate those subspaces into exact quadratic weight terms.
6. **Late-MLP compact programs:** use the newly found shared input cores only as candidates. Promote them to circuits
   only if they predict a named behavior and support selective extraction/removal beyond held-out CE preservation.

## Ranked actions now

1. Add the two final completed-package hard-abort regressions, rerun all owner/adapter/live historical checks, and
   freeze the five exact R585 bytes.
2. Commit and push only those five files, then give a fresh critic the immutable commit and all earlier attacks. The
   prior BLOCK reviews are evidence, not approval.
3. Enqueue the hash-pinned adapter only after explicit APPROVE. FIT controls whether SELECT opens; FINAL and OOD remain
   closed, and the maximum price remains 690 forwards with zero backwards or updates.
4. Audit any result from raw evidence before interpreting it. Preserve a null without threshold changes.
5. Add the new evidence-derived-failure and hard-abort distinctions to the shared playbook/handoff.
6. Start the next two distinct-circuit wave on pending opener and numbered-list successor, while retaining R585 as the
   parent reference investigation.

The active action is step 1. No R585 result, receipt, or evidence directory exists, and no rank result has displaced the
behavior-level circuit program.
