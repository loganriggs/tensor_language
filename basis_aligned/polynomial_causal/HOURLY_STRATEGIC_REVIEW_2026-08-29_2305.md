# Hourly strategic review — 2026-08-29 23:05 UTC

## Bottom line

The repaired shipped-program control passed, so the MLP2 allocation result is real:
inside the current all-table 36-site program, reducing only MLP2's table rank from 768
to 128 costs **0.000248 nat pooled** (`t=8.97`, 92,160 token losses) while saving
**4.205 million stored values**. Across the three document roles the costs are
`0.000205 / 0.000322 / 0.000190` nat.

This is a useful executable compression result, but it does **not** explain native
MLP2. In a partially compiled program with live downstream attention, MLP2 table
content was worth about `1.408` nat. The contrast is direct evidence that the relevant
notion of simplicity is conditional on the consumers that remain live.

A fresh 104.3-second test then rejected the tempting generalization that every shipped
MLP table can be collapsed independently. MLP2 and MLP3 mean rows are almost free,
but replacing MLP12 by its mean costs `0.02687--0.02913` nat. Replacing all three
together costs `0.03198--0.03522` nat, with a positive three-way excess of
`0.00588--0.00624` nat over the sum of single replacements. Exact CE additivity is
false. That test's intended inert control also failed, so its values are scoped
discovery/failure evidence, not certification.

## What fraction is actually explained

The strict whole-model balance sheet does not move:

- structural intervention coverage: **36/36 sites**;
- certified stored-parameter removal: **5.348245316%**;
- named causal cross-entropy: **10.923302467%**;
- unexplained causal cross-entropy: **4.72714 nat / 89.077%**;
- end-to-end extracted, selectively removable, OOD-certified actions: **0/68**.

Thus we have broad instrumentation and a new local allocation win, but only about 11%
of causal predictive behavior has a named mechanism. The rank-128 result stays out of
the strict storage ledger until the whole per-site allocation is priced and validated
under its governing certification rules.

## What the new computations mean

For two table ranks, the paired quantity was each token's loss under rank 128 minus its
loss under rank 768. Its pooled mean is `+0.0002484437` nat with standard error
`0.0000276838`; the ratio is the reported `t=8.974`. The effect is statistically real
but tiny.

For the mean-row experiment, define

$$
d_S=CE(\text{mean rows at sites }S)-CE(\text{full shipped program}).
$$

The measured interaction is the Möbius-style remainder

$$
I_{2,3,12}=d_{\{2,3,12\}}-d_{\{2\}}-d_{\{3\}}-d_{\{12\}}.
$$

It is `0.005881 / 0.006240 / 0.006143` nat. Even though fixed table writes add in the
residual stream, final RMSNorm and cross-entropy are nonlinear, so zero interaction was
never a theorem. More importantly, MLP12's large single-site cost shows that MLP2's
near-zero content value is site-specific, not a property of all compiled MLPs.

## Largest remaining gaps

1. **Consumer-conditioned MLP2 geometry.** We do not know which MLP2 errors matter
   when native attention remains live, nor why raw local MSE misses them.
2. **Per-site allocation.** One site admits rank 128, but the 18 MLP and 18 attention
   ranks have not been allocated jointly at equal literal price.
3. **Finite composition.** MLP0-C512 and rank-512 MLP2 retain a replicated positive
   interaction; MLP1 has not been added to the same controlled factorial.
4. **Late semantic consumers.** Copy and attention-interface probes exist, but
   capitalization, numeric formatting, syntax, and entity consumers are not a verified
   sufficient/necessary/OOD bank.
5. **Executable causal quotient.** No learned equivalence class yet predicts withheld
   consumers and unseen intervention compositions while supporting selective edits.
6. **Terminal closure.** No one of the 68 proposed actions meets extraction, removal,
   collateral, OOD, and receipt requirements end to end.

## Candidate actions and pruning

| Candidate | Information/causal value | Composability and falsifier | Cost/redundancy ruling |
|---|---|---|---|
| Finite error-Rayleigh validity pilot at MLP2 | Directly tests the missing downstream metric | Must predict held-out finite C512×MLP2 interactions | **Keep; highest priority** |
| Per-site shipped rank allocation | Converts the MLP2 win into literal storage savings | Sweep each site, then validate joint allocation | **Keep; cheap and executable** |
| Direct mixed-functional factorization | Targets the replicated non-additive term | Must predict held-out finite interactions | **Keep after metric pilot** |
| C512×MLP1×MLP2 triangle | Tests whether the failure is a shared early interface | Full factorial with equal-price controls | **Keep, moderate GPU cost** |
| Late-consumer quotient/Hankel realization | Best semantic/editability route | Withhold one consumer and one composition | **Keep, but consumer bank incomplete** |
| More raw CP/Tucker/HOSVD or unweighted MSE | Repeats a rejected objective; global rank 512 is algebraically impossible | Local reconstruction cannot license composition | **Prune** |
| Sparse document gate | Interaction is diffuse over roughly 108--118 effective documents | Frozen concentration test already failed | **Prune** |
| Exact additivity from fixed writes | Confuses additive residual writes with nonlinear final loss | Fresh three-site test is non-additive | **Prune** |
| Reopen Family F or expired checklist cells | Family F has a receipt-backed fit failure and the eight-hour audit is historical | Only new grammars can reopen successors | **Prune** |

## Ranked top five

1. **Run the finite MLP2 error-Rayleigh validity pilot.** It has the highest expected
   information gain because it distinguishes a wrong metric from finite nonlinear
   composition before another expensive fit. The exact prospective protocol is in
   `MLP2_ERROR_RAYLEIGH_VALIDITY_PILOT_PREREGISTRATION.md`.
2. **Measure an 18-site shipped-program rank-allocation frontier.** The MLP2 rank-128
   win is large in storage/price and the MLP12 failure proves ranks must be site-specific.
3. **Fit the mixed intervention functional only if the validity pilot predicts finite
   effects.** The shared 91.2% document mode makes this plausible, but not yet licensed.
4. **Run a controlled C512×MLP1×MLP2 factorial.** This tests whole-program composition
   and whether MLP1 is part of the same interface failure.
5. **Expand verified late consumers, then fit a causal quotient and finite Hankel
   state.** This has the best semantic/editing payoff but depends on more independently
   validated consumers.

## Executed action

The highest-priority outstanding recovery from the previous review completed in
**92.9 seconds** with all controls passing. It established the rank-128 MLP2 result
above. The subsequent 104.3-second three-site experiment then falsified broad
site-independence and exact additivity, while preserving its repeated inert-control
failure. These are numerical outcomes, not runners or plans.

The next safe action has also been frozen prospectively: a 64-document, two-background,
three-program finite error-Rayleigh pilot. It measures full-sequence error fields,
symmetric finite JVPs, exact categorical-Fisher logit response, separate attention-5/6
responses, deranged/random controls, and one-shot held-out prediction. Passing it may
authorize a fresh weighted rank-512 fit; it cannot by itself certify low tensor rank.

## Eight-hour-plan audit status

The 12:00 UTC window expired eleven hours ago. Its receipt-backed record remains:
Family F is a fit negative; E1.1/E1.3, E2.1/E2.2, E3.1/E3.2 are measured negatives;
E1.2, E2.3, and E3.3 were prospectively pruned; E4.1 has a failed terminal-copy screen;
E4.2/E4.3 have no completed outcome. No checkbox, cache, or unrun code is being counted
as current progress.
