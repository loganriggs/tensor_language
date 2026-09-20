# Semantic source-port census

## Question and baseline

The A grouping combines direct embedding recurrence with early computed writes. The v696/v697 semantic controls show number evidence at both subject and attractor positions. This screen asks whether their sign roles come from the recurrence alone or from computed writes. Earlier subject-only source censuses do not answer this paired-role question.

Frozen inputs: the 48 opposite-number rows in SUBJECT_ATTRACTOR_CONTROL_V696_ROWS.json and their 48 matching-number counterparts in SUBJECT_CONGRUENT_ATTRACTOR_V697_ROWS.json. No new fitting or fresh-text claim. Individually edit the five named ports and replay A=[0,1], B=[2,3,4] at each noun site; keep baseline cached first values and embedding residual reinjection. This is a selected port/site intervention, not an entire embedding edit.

## Exact direct path

Let block l mix its state as alpha_l*x + beta_l*x0. The portion attributed solely to initial embedding recurrence is e_l=c_l*x0, where c starts at1 and c <- alpha_l*c+beta_l at each block. At the pre-attention11 boundary, perform twelve recurrences, l=0,...,11. Equivalently c is the product of all alpha terms plus the sum of beta_l times every subsequent alpha. Thus its edited source delta is c*(edited_x0-x0). This linear transport is exact over real arithmetic; normalization and the norm-preserving number edit remain explicit. Native float32 accumulation creates a measurable numerical discrepancy, tested against the captured port at relative1e-5.

The coefficient requires24 native scalar lambda values to compile and one scalar to execute. Token embedding lookup, initial RMS normalization and the frozen decoder are additional dependencies. Closing this direct port does not close other source ports, attention, suffix context or the full model. No standalone simplification is inferred.

## Registered gates

1. Instrument: old A/B effects replay within absolute1e-4; direct-port formula within relative1e-5; norm preservation1e-5; exactly24 prefix and120 suffix batch calls.
2. Direct-port sufficiency: its number effect predicts A within relative10% for every panel, role and family.
3. Computed-port screen: at least one computed port has subject-removal damage on75% of rows in both panels, matching-attractor damage on75%, and opposite-attractor damage on at most25%. Positive damage threshold1e-4.

Report individual-port additive prediction of A/B as the simple composition baseline, with every context cell retained. Port selection is an opened-data screen; passing it does not identify a unique semantic unit. All full model weights and native generators remain charged. Nulls and numerical invalidity are recorded separately.

Runner: ../bilinear_quotient/ops/run_semantic_port_census_v1.py. Result: ../bilinear_quotient/circuits/followups/semantic_port_census_v1_result.json (pending at registration).

## Native results

Instrument passes: grouped A/B absolute replay discrepancy1.62e-5; recurrence relative discrepancy4.45e-7. Compiled recurrence coefficient55.38656468. Direct-port sufficiency fails (worst34.62%). Early writes0–3 and middle writes4–7 pass the computed signed-role screen; MLP8 and MLP10 fail its full conjunction. Direct recurrence also exhibits the sign pattern, so computed-port success does not imply uniqueness.

Individual source effects add well for A (worst4.54%), but B misses by27.78%. This motivates measuring all three pairwise B interventions on the same opened panels. Pair reconstruction uses effects E_i and E_ij: J_ij=E_ij-E_i-E_j, prediction=sum(E_i)+sum(J_ij). The remainder E_234-prediction is the exact third-order Boolean interaction for these finite interventions. It is not the polynomial degree of the normalized model. No fitting is involved, but obtaining these native effects still requires multiple model executions; no compressed predictor is claimed.

Pair runner semantic_port_pairs_v1 registers all-pair relative10% closure and a stronger single-shared-pair10% gate. Literal cost24 prefix/168 suffix batch calls; baseline/error semantics unchanged.

## Pair and fresh transfer results

Opened all-pair effect reconstruction passes, max1.8977%; the registered single-shared-pair gate fails. Exhaustive CPU subset scoring finds no two-pair subset below10% across all opened cells: best pair23+pair34 is10.00645%, a strict failure. Primitive effects replay bit-exactly across the single-port and pair runs. A planted Boolean polynomial checks interaction signs and third-order remainder extraction independently.

Freeze pair23+pair34 and all three pairs before testing four new constructions, 48 opposite-number and48 matching-number rows (same six-noun vocabulary). All-pair effect error remains1.9821%, versus38.11697% for the additive baseline. Frozen two-pair error is18.6842%, so its prediction fails. The all-pair registered conjunction also fails: baseline native agreement capability falls to50% and83.33% in the opposite/congruent earlier_noticed plural cells, below90%. Other cells pass native capability. Both failures remain recorded. An alternative two-pair subset passes on this now-opened panel but is not promoted or substituted for the registered choice.

The fresh instrument compares the captured-prefix suffix against an independent full native forward (exact margin agreement), rather than the old-panel A/B replay. Its inherited prediction-description string still says grouped A/B replay; the executed check and fresh preregistration are the full native baseline comparison. Calls are24prefix+168suffix+8full-model controls. Frozen row hashes are in SEMANTIC_PORT_FRESH_ROW_AUDIT.json.

[Independent pair audit](SEMANTIC_PAIR_GRAPH_CPU_AUDIT.json) verifies planted signs, primitive replay, subset bars and arithmetic. [Native capability audit](SEMANTIC_NATIVE_CAPABILITY_CPU_AUDIT.json) confirms subject/attractor positions, answer token ordering and number labels, and lists four incorrect native rows without filtering them. This is not evidence of a scoring bug. Predicting the native causal effect on those rows is distinct from predicting correct language behavior.

## Status and next handoff

The full B pair graph is a transferable description of finite intervention effects; single-pair sparsity and frozen two-pair sparsity are falsified on these panels. The retained source groups remain wide and native-module-based; their learned features are not identified. Evaluating all source singles and pairs costs six edited suffix evaluations, so this is not computational compression versus one native combined intervention. Native source-generation and suffix contexts remain open. No standalone extraction, selective unrelated-behavior preservation or five-property adoption is claimed.

Next circuit question: do the needed pair interactions arise in block11 before the post11 interface or in the later suffix? Compare actual joint pre11 edits against sums of separately computed post11 responses. This distinguishes source-boundary interactions from downstream composition and supplies concrete terms for the next weight-folding hour's backward quadratic observable baseline. The last CPU audit has already localized capability failures and measured signed interaction support without changing the frozen graph.
