# Adversarial review findings (subagent red-team, 2026-07-29) — ACCEPTED

Eleven findings against the program-substitution claims; program's response in brackets.

HIGH:
1. The 89.4% ledger sums superadditive single-interface floors; the all-programs-joint substitution
   was never run (every measured joint case is superadditive: MLP0+1 +18%; attn stack 2.4x).
   [FIX QUEUED: qk_joint_mlp_stack.py — all credited MLP programs substituted simultaneously +
   joint floor as denominator.]
2. Headline dominated by weighting: MLP0+1 = 58% of weight at ~97%; UNWEIGHTED mean across the 36
   interfaces = 59%. [ACCEPTED: headline now dual-reported.]
3. Attention credit (1.67 nats) is ridge reconstructibility; the random-basis null achieves 96% of
   it. [ACCEPTED: attention credit deflated to the sym-vs-rand margin in v4 until predicate-style
   programs exist per layer; booked separately.]
4. Induction predicate/finetune templates+scalars were fit on the same 48 prefixes the flagship
   numbers are scored on; METHODS' "all audits held-out" was false for this result; "3 scalars/head"
   ignored the ~8k-entry fitted position template; template only valid at the fitted window length.
   [FIX QUEUED: qk_induction_refit_heldout.py — fit on cooc prefixes, evaluate on fresh FineWeb and
   Pile prefixes at multiple periods/window lengths. "Code beats model 116%" reframed: task-metric
   optimization scored near fitting data, not fidelity.]
MED:
5. Ledger numerator and denominator computed on different audit sets/lengths. [v4 will recompute
   all costs and floors in one script on one audit.]
6. No random-A/trained-U control for the quadratic programs; no polish-on-null control; polish
   table-blend scalars 0.30-0.57 partially re-specify the programs. [FIX QUEUED:
   qk_random_feature_control.py at MLP0 and MLP17.]
7. Only the induction predicate ever passed the code-verify meaning gate -> ~7.9 of 8.90 explained
   nats are anonymous fitted surrogates. [ACCEPTED: metric renamed SUBSTITUTABLE fraction; a second
   'meaning-verified' column gates on code-verify passes.]
LOW:
8. Hardcoded credits (attn1 = 0.99 from an old arc; v3 has no generating script). [v4: zero
   hardcodes, attn1 re-audited uniformly.]
9. Arm selection on the reporting audit. [v4: arm selection moves to a cooc-held-out split.]
10. Pairpatch "VERIFIED" overstated (control wins the CE metric; support is MSE-at-covered-positions
    only). [Language corrected.]
11. Induction-service checks: single 48-seq eval, no variance, no threshold; one LOG range misquoted.
    [Threshold + second prefix set to be added; misquote corrected.]

Clean cross-checks by the reviewer: v3 arithmetic internally exact; all quoted program/polish numbers
match their JSONs; cooc vs FineWeb document-disjoint by construction.
