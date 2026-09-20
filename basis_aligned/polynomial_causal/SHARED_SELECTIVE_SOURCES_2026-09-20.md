# Shared source selection fails beyond per-input optimization

A12coefficient role-level selector was fitted only on opposite-panel first derivatives, with8controlconstraints and signed target retention. Its certified fit ceiling is47.50%for subjects and0.1918%for attractors at the registered8%reference-control budget, below80%target. Primalviolations anddual gaps<=5.56e-17. These bounds apply only to this six-port,box-bounded,first-order sharedprogram; they are not lower bounds against general circuits.

Native sharedselector passes0/16cells (0/8heldcongruent), failingretention16/16 andcollateral10/16. The independentlyoptimized per-input LP ceiling passes11/16nativecells:5/8opposite,6/8congruent. Its fivefailures allfailretention; twoalsofailcollateral. UnitBpasses4/16. This oracle uses eachinput's native derivatives/newcontrols, so its higherpasscountis not a reusable extraction orheldoutsuccess.

Instrumentation passes: oldgradient/nativeoutputreplay4.44e-16, fixed-directionfinite-difference worst7.10e-8relative. Counts12prefix24double72reverse gradients28native suffixes,4.55s. Planted independent-control andcoupled-control LPs recover retention1and.08, respectively, withzerodualitygap. Negative results are not explained by the solver failing its specified problem.

Executed CPU structural-sharing audit: allow onefixed6coefficient rule perrole/template (24total). Fitretentions subject/unlike103.56%,subject/beside48.18%,attractor/unlike5.90%,attractor/beside0.1918%. Noneof4groupspasses held derivative80%retention/8%referencecontrol gate. No newweights were nativelyevaluated; the unlike-subjectgroup's heldcontrolratio10.83%slightlyexceeds thisgate. Splitting only bysentence structuredoesnotresolvesharing.

Interpretation: bothsharedsourcegeometry and per-inputtarget/controlcoupling constrainthisinterface. Do not keepaddingcase-specific scalarcoefficients andcallthatsharedfeatures. The next circuit route should expose richer, explicitlypriced residualfeatures or split the aggregated native sourcewrites, then compare causal selectivity and reuse against thesesix-port baselines. Native prefix/sourcegenerators andsuffix remainrequiredbyallcurrentarms. A broader controllable space is not automatically a semanticcircuit; requirefrozenfeaturedefinition andheldoutcausalcontrols.

The fullgoal remainsunmet: stable sharedfeatures, independentlyexecutable extraction, broaderselectivity andprospectivecomposition remainopen. Exactlocalfolds andnativeaccountingresults stayvalid but do notmeetthoseconditions.

Receipts: ../bilinear_quotient/circuits/followups/shared_selective_sources_v1_result.json; SHARED_SOURCE_GROUPING_AUDIT_V1.json; SHARED_SELECTIVE_SOURCE_LP_CONTROL.json.
