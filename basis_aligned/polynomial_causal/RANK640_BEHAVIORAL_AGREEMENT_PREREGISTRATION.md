# Preregistration: rank640 behavioral agreement beyond cross-entropy

Date: 2026-08-28

Status: frozen before the first top-1, KL, or frequency-stratified rank640 outcome.
The two cross-task rows and rank were previously opened for CE and causal validation;
this is a new-instrument audit of the admitted candidate, not fresh OOD promotion.

## Question

The context-free program retained a nontrivial CE fraction while agreeing with the
live model's top token on less than one quarter of positions and failing almost
completely on rare targets. Does the admitted 516,707,766-value rank640 complete
program avoid that failure?

Rebuild the exact rank640 program using only the original skip80 fit rows. On the
already-authorized skip31000 and skip35000 roles, jointly score native and program
logits at positions 64--255. Report:

- live/program cross-entropy and the difference;
- $\mathrm{KL}(p_{\rm live}\|p_{\rm program})$;
- top-1 agreement;
- live and program top-1 accuracy;
- the same quantities in true-target fit-frequency buckets
  $0$, $1$--$4$, $5$--$24$, $25$--$124$, and $125+$.

Fit frequency counts true next-token occurrences in the 480 fit rows at the identical
192 scored positions. It is not current-token coverage.

## Frozen gates

1. Exact 516,707,766-value ownership, no native calls/tables/fallback, disjoint
   checkpoint storage, and exact predictive/causal parent hashes.
2. Recomputed native and rank640 CE must match the admitted parent within $2\times
   10^{-6}$ nat.
3. Overall top-1 agreement is at least 0.98 on both roles.
4. Program top-1 accuracy is no more than 0.005 absolute below live on either role.
5. Mean $\mathrm{KL}(p_{\rm live}\|p_{\rm program})\le0.01$ nat on each role.
6. On the combined target-frequency-0--4 tail, program accuracy is no more than 0.005
   absolute below live and retains at least 97% of live accuracy on both roles.

The result is create-only and source-closed. A failure does not revoke the earlier CE
or intervention receipts; it says the current definition of admitted simplicity needs
the failed behavioral instrument added to its constraint set.

