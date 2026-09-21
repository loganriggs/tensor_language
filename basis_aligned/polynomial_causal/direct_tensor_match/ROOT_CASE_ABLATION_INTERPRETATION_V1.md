**The shared feature affects case preference, but selective manipulation and native response fidelity fail.**

Managed runtime1.71s, numerical controls PASS: native quartic replay4.73e-8, normalized reader/writer pairing2.38e-7 from1, exact zeroedit. No fit. Remove native rank-one projected quartic component or its learned approximation from the original final residual with the same actual MLP17 denominator; retain all other terms. Final normalization/softcap explicit. The native projection is defined by the learned writer; this is an operational reference, not a uniquely identified native semantic unit.

Case score is logsumexp over8805 uppercase-initial token variants minus logsumexp over their8805 lowercase-initial counterparts, with matched remaining bytes and optional leading space. It is a vocabulary-class preference, not task accuracy. Non-newline positions are comparison conditions, not inherently unrelated behavior.

|Domain/strength|Native case effect afternewline|Native effect elsewhere|Newline/nonnewline mean-absolute ratio|Approximation logit-response error newline/other|
|---|---:|---:|---:|---:|
|FineWeb .25|-.0997|+.0956|1.691|26.08/24.43%|
|FineWeb 1|-.6333|+.3956|1.980|31.07/29.46%|
|Code .25|+.00268|+.1400|.0649|25.94/11.69%|
|Code 1|-.00045|+.5278|.0712|26.16/12.08%|

Counts106newline/3718other FineWeb,210/3614code, all16prefixes each. Registered10%fidelityFAIL everycell. FineWebdirection/magnitude subcriterion passes, but the combined selectivity criterion requires ratio>=2 at BOTH strengths and FAILS. Do not round1.980up to2. Codeeffects were descriptive, not partofFineWebdirectiongate. Original fullgraph/reader-transfer failures remain.

Actual CPU successor audit: FineWebnewline caseeffectnegative13/16prefixes at.25,14/16at1; elsewherepositive16/16atboth. Removing anysingleprefix leaves.25selectivity1.625–1.823, stillfails. Full-strength ratio1.818–2.179 variesacross exclusions, so its near-threshold result is not robust. Code newlineeffectnegative6/16and9/16, whileelsewherepositive16/16. Theseareprefix sensitivities, not independent-confidenceintervals.

Scientific consequence: top activation conditions and veryhighnewlineAUC did not imply a newline-selective causal computation. There is a real output-class effect, whose direction depends on context and whose learned removal is inaccurate. Keep this as a failed promotion screen and a coupled case-effect hypothesis, not a discovered capitalization circuit. No adoption, untouchedOOD, stablefeatureidentity, extraction or composedreuse established.

Next discriminating question is whether the learned feature's scalar error or downstream nonlinear sensitivity explains native removal failure. The alreadyavailable matched-reader test had15.3%contextresponseerror, but this screen uses different longercontexts and absolutecomponent removal; do not ascribe the gap to nonlinearities without a matched measurement. A matched scalar-versus-native-logit error decomposition on the SAME captures would distinguish these before altering graph capacity or relabeling semantics.

[Registered plan](ROOT_CASE_ABLATION_PLAN_V1.md) · [Native results](ROOT_CASE_ABLATION_V1.json) · [Prefix audit](ROOT_CASE_DOCUMENT_AUDIT_V1.json) · [Runner](../../bilinear_quotient/ops/run_root_case_ablation_v1.py).
