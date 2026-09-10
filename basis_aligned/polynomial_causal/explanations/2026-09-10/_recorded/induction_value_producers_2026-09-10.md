# The equality-copy message needs both value producers in this screen

September 10, 2026. **Neither tested value branch reproduces the full equality
message's causal effect. No new independently extracted circuit is established.**
The run is mechanically valid, and a separate saved-output audit identifies an
avoidable error in the screen's capability/materiality specification.

After the [shared-memory mathematics](shared_computation_and_editable_memory_2026-09-10.md),
we returned to a trained-model computation with existing evidence: four induction
heads that fetch the token following an earlier occurrence of the query token.
Earlier work demonstrated extraction and OOD effect prediction for their equality
service, but failed the stronger induction-only collateral guarantee. Those
results remain as recorded in the
[existing induction dossier](../../bilinear_quotient/circuits/campaign_2026_08_30/02_induction_copy.md).
The old R593 donor-factor interchange remains instrument-invalid and unresolved;
this new experiment tests value-producer removals instead.

## What was computed

Each selected attention head mixes two value vectors:

\[
v=(1-\lambda)v_{\rm local}+\lambda v_{\rm first}.
\]

The local value comes from that layer's current input. The shared first-layer
value is produced once and passed to later heads. A learned scalar `lambda`
sets the mixture and need not lie between zero and one.

For the matching payload source `s` and query `q`, the head writes `p(q,s) v_s`,
followed by its output matrix. The coefficient `p` is the product of the two
native normalized and position-rotated QK dot products, each divided by128.
We removed its shared-first contribution, local contribution, or complete mixed
contribution. The query, source projections and downstream layers recomputed
during each run. Summing both branch removals was an implementation check;
adding their final behavioral effects was a separate scientific test.

The fixed sites were zero-indexed L5H5, L7H3, L8H3 and L8H4. The screen used all
72 existing FIT worlds, each with four combinations of queried source and payload
assignment. An equal-count neutral-source removal supplied a contrast. SELECT,
FINAL_TEST and OOD data were excluded. This is not fresh generalization evidence.

## Trained-model outcome

Relative error here means the norm of the difference between a branch-removal
effect and the full-removal effect, divided by the norm of the full effect.
Full-vocabulary vectors are centered across vocabulary before comparison.
Each number aggregates all72worlds in one condition; ranges cover four conditions.

| Measurement | Observed range | Registered maximum |
|---|---:|---:|
| Shared-first branch: full-vocabulary effect error | 88.53–92.02% | 10% |
| Local branch: full-vocabulary effect error | 35.16–38.61% | 10% |
| Shared-first branch: answer-versus-other margin error | 36.54–45.89% | 10% |
| Local branch: answer-versus-other margin error | 71.71–76.70% | 10% |
| Sum of branch effects: full-vocabulary error | 11.82–19.03% | 10% |

The smaller-error branch depends on the readout: shared-first is closer on the
task margin, local on the full output vector. Neither meets its frozen criterion.
Do not promote one by discarding the readout on which it performs worse. The
composition miss concerns downstream effects; the local value mixture itself
reconstructs accurately.

Execution used63forwards and2016sequences, with162attention transactions, in6.068
executor seconds. Native versus zero-edit outputs agree exactly. Independent full
and summed-branch removals agree within2.146e-5 absolute logit error and7.453e-7
relative error. The value-mixture relative discrepancy is at most3.635e-8.
Untouched head/query outputs and the shared first-value stream stay bitwise equal;
all hooks are restored. All545902902 native parameters remain, with zero saving.

[Registered protocol](../INDUCTION_VALUE_PRODUCER_SPLIT_V1_PREREGISTRATION.md),
[native result](../INDUCTION_VALUE_PRODUCER_SPLIT_V1_RESULT.json), and
[seven CPU controls](../INDUCTION_VALUE_PRODUCER_SPLIT_V1_CONTROLS.json).

## Specification error and its correction

I registered an absolute mean gold-probability loss of at least0.10, but native
gold probability averages only0.0336–0.0374. The expected answer is preferred over
the competing payload on most rows while still receiving little mass among all
50,304 vocabulary entries. Pairwise accuracy and full-vocabulary probability are
different measurements.

Since edited probabilities cannot be negative,

\[
\operatorname{mean}(P_{\rm native}-P_{\rm edited})
\leq\operatorname{mean}(P_{\rm native}).
\]

The0.10 loss threshold was therefore unattainable. The prior R586 receipt already
contained enough information to establish this ceiling. It also used a0.75
pairwise-accuracy floor, whereas this screen newly required0.85. Its weakest
condition was already0.7778 in R586 and remains exactly that now. There is no
contradiction with the older passed capability result.

The [executed contract audit](../INDUCTION_PROBABILITY_CONTRACT_AUDIT_V1_RESULT.json)
matches all288 FIT core cases to the old receipt. Correctness fractions agree
exactly, prior/current mean probabilities differ by at most2.32e-8, and margin
replay differs by at most1.77e-5. It adds a small reusable
[probability-loss feasibility check](../induction_probability_contract_audit_v1.py)
for future specifications. The failed experiment is preserved without rescoring,
lowering its thresholds, or calling the check a circuit discovery.

For interpretation only, complete equality removal reduces mean gold probability
by93.85–95.94% relative to the native mean. Thus failure of the impossible absolute
bar does not establish weak causal involvement. This descriptive ratio is not a
replacement pass criterion. The independently measured branch-fidelity errors
still reject either producer as a sufficient stand-alone explanation at this
interface. The complete coupled operation remains relevant; its independently
generated semantic inputs, selective reuse, and structural simplification remain
the research questions. No head, dose, rank or condition rescue is licensed.
