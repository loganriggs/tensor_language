# Hourly circuit and systems review — 2026-09-04 18:03 UTC

## What a useful circuit must eventually provide

The goal is a predictive, composable, manipulable, and simple account of the model as a tensor program. A useful circuit must:

1. say what information it reads, what computation it performs, what it writes, and who reads that result;
2. group pieces across native heads/MLPs when they implement one variable, and split a native module when it serves different tasks;
3. predict held-out and shifted examples;
4. reproduce the target computation or signed causal effect as an executable extracted circuit;
5. support selective removal, interchange, and editing without damaging unrelated behavior, including redundancy and interaction;
6. compose predictably with reused and task-specific parts; and
7. be stable across data splits, gauges, and fitting starts, or be defined by downstream operational equivalence.

Storage, parameter count, tensor rank, and reconstruction can price or control a result. They cannot by themselves discover or
interpret a circuit.

## Repository-timestamp audit since 17:03

Five new GPU screens reached terminals in 9.3, 8.1, 7.1, 4.8, and 7.0 serial minutes; their median is 7.1 minutes. A sixth exact
MLP15-by-MLP17 interaction was derived from immutable evidence in about three claim-to-result minutes with zero model calls. The
shared latency tool reports a 7.6-minute overall median across the current thirteen-screen window, within the ten-minute target.
Actual compute for these screens was 0.73–0.86 seconds each; reasoning, authoring, validation, and queue polling remain the dominant
costs.

Scientific progress in that sequence:

- later attention is negligible in the response to the head-11.3 agreement counterfactual; the downstream change is mainly through
  MLPs;
- MLP15+17 reproduce the MLP15–17 response within 11.6% relative error, while MLP16 and its interaction are each below 0.6% RMS;
- MLP15 and MLP17 both contribute and combine nearly additively: their interaction is 0.655% RMS;
- these effects mainly compensate plural-to-singular head swaps, so they are not yet established as a bidirectional subject-number
  reader.

The deep head-11.3 projector used 1,206 forwards and 902 backwards per complete run. Two pre-fit implementation failures and one
artifact-only republication exposed the hour's largest avoidable cost. The final scientific terminal is still instrument invalid:
all nine fits improved their objective, but by only 0.0254–0.0469 against the registered 0.05 health bar. It is not a subspace null.

Systems changes completed this hour:

- exact factorial corners are reused instead of recomputed;
- atomic claims and prior-art receipts prevented duplicate module screens;
- Task 14 and its negative results were added to the canonical circuit dossier;
- failed-run logs are now copied aside before retries overwrite them;
- snapshot-class and serialized-frame hash bugs have regression tests;
- the corrected projector receipt/bundle passed independent audit.

Claude remains circuit-only and reports the same 7.6-minute median. The last-60 execution failure rate is 13.3%, with 35 minutes
between failures and retries. The new preserved failure logs should make the next engineering review rank causes rather than infer
them from overwritten traces.

## Step back: is the current path still highest-information?

Module grouping remains useful, but fitting another small subspace immediately is not. The proposed MLP15/17 product-space fit was
stopped before execution because its full-rank causal object is not yet characterized. The current evidence supports a narrow,
direction-dependent compensatory response. Compressing that response before testing its syntax/lexical stability could produce a
small but semantically misleading feature.

The product coordinates also have a large invisible part: the MLP output matrix maps 4,608 product coordinates to 1,152 residual
coordinates, so its kernel has dimension at least 3,456. A projector can move inside that kernel without changing model behavior.
Any eventual product-space claim must therefore be about the downstream-visible causal function or compiled quadratic tensor, not a
unique list of hidden product directions.

## Confounds that must be closed before fitting

- Define removal and sufficiency with opposite signs from one explicit donor-minus-base response.
- Classify every relation by its frozen `expected_relation` and arm; A/P/C family letters do not determine target/control status.
- Measure paired, cross-noun, literal cross-syntax, and complete-subject rows before calling the response syntax-general.
- Use depth-ordered endogenous execution for the MLP15+17 circuit: alter MLP15, recompute MLP17, then alter MLP17. A cached factorial
  is attribution evidence, not an executable composition test.
- Require an unrelated behavior to have a live nonzero MLP15/17 effect before using a ratio as a collateral-damage control.
- Keep the outer validation set physically unopened and avoid choosing bars after viewing SELECT.
- Require optimizer improvement before interpreting target/control performance; the completed projector demonstrates why.

## Ranked next moves and kill conditions

1. **Full-rank MLP15/17 conditional-response panel.** Measure exact reset and rescue for each module and their depth-ordered joint
   response across every frozen relation type. Expand each bilinear response into base-by-change, change-by-base, and change-by-change
   terms. This advances computational specification, grouping/splitting, composition, and selective intervention. Kill a general
   agreement-reader interpretation if the full response is not stable across directions, lexical groups, and syntax; retain a
   narrower compensatory-path claim if that is what passes.
2. **Downstream-visible causal response quotient.** If the panel passes, identify operationally equivalent output functions first,
   quotienting product directions that the output matrix cannot distinguish. This advances stable identification and extraction.
   Kill it if removal and sufficiency cannot agree on held-out rows or if geometries vary without stable causal responses.
3. **Product-space feature and exact quadratic compilation.** Only after (1), fit the task-conditioned product response, require
   removal, sufficiency, unrelated live controls, and executable MLP15+17 composition, then compile it exactly into quadratic weights.
   Kill the product interpretation if output space passes but no downstream-visible product quotient does through rank 8.
4. **Direct bilinear term hypothesis.** Before optimizing a free projector, test whether the response is explained by a small number
   of exact base-by-change or change-by-change quadratic terms. This is a genuinely different mathematical route and may give a named
   computation directly. Kill it if held-out causal effects require many unstable terms or do not compose across MLP15 and MLP17.

The active concrete step is (1). It is a causal circuit screen, not a rank sweep. The product-subspace claim stays active but blocked
from execution until this full-rank precursor and the red-team's sign/role/gauge requirements are frozen and pass.
