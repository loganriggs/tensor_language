# Strategic review — 2026-08-29 21:10 UTC

## Outcome first

The first fresh-data composition test of two independently compressed MLPs is
complete. MLP0-C512 is nearly lossless by itself, and MLP2-FULL512 retains the
large improvement seen in its first evaluation. When installed together they do
**mostly, but not cleanly, compose**: the combined extra cross-entropy is
`0.064996` nat rather than the additive prediction `0.056257` nat.

The excess

$$
I = \Delta CE_{\mathrm{BOTH}}
  - \Delta CE_{\mathrm{C512}}
  - \Delta CE_{\mathrm{FULL512}}
  = 0.008739\ \text{nat}
$$

has a document-bootstrap 95% interval `[0.007511, 0.010014]`. Thus the excess is
precisely positive on this sample. The preregistered formal label is nevertheless
`interaction_inconclusive`: clean composition required the combined arm to be no
more than `0.01` nat worse than FULL512, while it was `0.012120` worse; decisive
incompatibility required the *lower* interaction bound to exceed `0.01`, while it
was `0.007511`. We preserve those thresholds rather than relabeling after seeing
the answer.

## What was computed

The experiment ran four physical whole-model interventions on the same 192 fresh
FineWeb documents:

- `NATIVE`: neither replacement is installed.
- `C512`: only the frozen rank-512 MLP0 program is installed.
- `FULL512`: only the frozen rank-512 free-factor MLP2 program is installed.
- `BOTH`: both frozen programs are installed simultaneously.

For an arm $a$, extra cross-entropy is

$$
\Delta CE_a = CE_a-CE_{\mathrm{NATIVE}}.
$$

It measures how much worse that intervention predicts the actual next tokens. A
positive factorial interaction $I$ means the joint damage is greater than the
sum of the standalone damages: each program changes the input distribution seen
by the other enough that independently good fits cease to be independent.

| arm | extra CE (nat) | teacher KL | centered-logit NRMSE | native top-1 agreement | task accuracy |
|---|---:|---:|---:|---:|---:|
| `NATIVE` | 0 | 0 | 0 | 100.00% | 40.918% |
| `C512` | 0.003381 | 0.004478 | 0.04070 | 96.08% | 40.801% |
| `FULL512` | 0.052876 | 0.055820 | 0.14030 | 86.74% | 40.199% |
| `BOTH` | 0.064996 | 0.067903 | 0.15453 | 85.76% | 40.015% |

`Teacher KL` compares the whole candidate and native probability distributions,
not just the observed correct token. `Centered-logit NRMSE` compares their vectors
of pre-softmax scores after subtracting each vector's mean, normalized by the
native score scale. `Top-1 agreement` asks how often they choose exactly the same
next token. `Task accuracy` asks how often each chooses the dataset's actual next
token.

The combined arm is only 0.903 percentage points below native task accuracy, but
it disagrees with native top-1 on 14.24% of positions. It is therefore much closer
to a compact useful predictor than to a faithful emulator.

The effect is stable across prefixes: combined extra CE is `0.05653`, `0.06024`,
and `0.06500` on 48, 96, and 192 documents. The corresponding FULL512 values are
`0.04535`, `0.04926`, and `0.05288`; C512 is `0.00224`, `0.00224`, and `0.00338`.

## What the interaction means

The MLP0 replacement's marginal cost is `0.003381` nat on the native trajectory,
but `0.012120` nat after FULL512 is installed: **3.59 times larger**. Conversely,
FULL512's cost is 16.5% larger after C512 is installed. Actual joint damage is
15.5% above the simple additive prediction.

This is evidence for a missing interface, not evidence that either program is
worthless. MLP2-FULL512 independently reproduces its earlier result on a second
fresh 192-document role (`0.05288` versus `0.05147` nat), while C512 remains very
accurate. The likely problem is that FULL512 was fitted on native pre-MLP2 states.
C512 changes those states slightly; the learned MLP2 factors are not robust to that
trajectory shift. A program fitted jointly on native and C512-produced states is
the cheapest direct test of this explanation.

## Time and infrastructure

- Freezing 192 evaluation plus 192 unopened training documents took about 23
  seconds.
- The final four-arm physical run took **20.29 seconds**.
- Two earlier attempts failed before scientific outcome access: one because
  lineage metadata mentioning an intentionally absent historical row was mistaken
  for a live row dependency, and one because an overbroad check treated a concurrent
  branch advance as a source change. Both failures are preserved.

The science now takes tens of seconds. The larger elapsed cost was provenance
hardening and independent adversarial review. The registry scanner now verifies an
embedded waiver proof before ignoring its historical missing path, and the recovery
binds exact source bytes rather than requiring the shared branch pointer never to
move. These are reusable fixes; this exact overhead should not recur.

## Honest whole-project balance sheet

- `36/36` intervention surfaces are addressable. This is structural access, not
  semantic understanding.
- `5.348245316%` of storage has a certified removable description.
- `10.923302467%` of the measured causal CE gap is named and recovered.
- `4.72714` nat, or `89.077%`, remains unexplained by that strict accounting.
- `0/68` terminal extraction/removal/OOD actions pass the complete standard.

The composition result does not move this ledger: it is in-distribution, its MLP2
parent is not yet certified, and it supplies no terminal edit or OOD claim. It does
locate a concrete source of non-composability and therefore changes what should be
fit next.

## How downstream circuits help explain early layers

Yes: learning additional late circuits is a high-return complementary entry point.
An early representation should not be clustered merely by Euclidean proximity. Two
early states $x,x'$ can be treated as equivalent relative to a verified consumer
bank $g_1,\ldots,g_m$ when

$$
(g_1(x),\ldots,g_m(x)) \approx (g_1(x'),\ldots,g_m(x')).
$$

This makes an early coordinate meaningful operationally: it is the information
needed by capitalization, number formatting, copying, syntax/entity, or another
named downstream computation. The bank can reveal shared directions used by many
consumers, private directions used by one, and a hierarchy/DAG in which reusable
features feed several later decisions. It also supplies a downstream-weighted error
metric for fitting MLP0/MLP1/MLP2 instead of unweighted activation MSE.

The danger is an incomplete or noncausal consumer bank. If a late probe correlates
with a behavior but is not sufficient/necessary, or if an important consumer is
missing, it can collapse early states that the model actually distinguishes. Each
consumer therefore needs held-out sufficiency, necessity, shuffle, off-target, and
OOD controls. The current copy-head bundle is only a partial consumer: it is causally
important, but its replacement still has too much collateral damage.

## Simplicity and utility

The two parents have a fixed executable grammar and price rather than a post-hoc
visual notion of simplicity: rank/product count, stored coefficients, dense matrix
multiplications, and native calls. Composition asks for a benefit that local
reconstruction cannot demonstrate: whether two small modules can be linked without
hidden access to native intermediate states. The positive interaction is exactly the
kind of failure that a useful simplicity definition must expose.

The next comparison must hold the MLP2 price fixed and change only what data/objective
it sees. If a joint-background fit reduces the interaction while keeping standalone
CE, KL, OOD, and cost fixed, it earns a genuinely more useful notion of simplicity:
the same-size program is more composable and hence more suitable for prediction,
extraction, and controlled edits.

## Pruned candidates

The following are currently poor return:

- longer optimization on native-trajectory local MSE without changing the objective;
- another native-channel subset sweep at rank 512;
- HOSVD or gauge balancing judged only by local reconstruction;
- simple document means/second moments as the missing context gate;
- composing many more uncertified modules before repairing the first measured
  interface.

They either duplicate negative work or cannot explain the measured interaction.

## Ranked next five

1. **Equal-price trajectory-robust MLP2 refit.** Fit the same 512-product program on
   both native and frozen-C512 pre-MLP2 states, with a downstream logit/Fisher loss;
   then repeat the four-arm test on fresh data. This directly predicts that the
   `0.008739` interaction and `0.012120` marginal penalty will shrink without buying
   more coefficients.
2. **Verified late-consumer bank.** Capitalization, numeric formatting, syntax/entity,
   and copy provide operational semantics and a task-relevant metric for early-layer
   factorization. This can improve both explanation and selective edits.
3. **Shared-plus-private or conditional block factors across the two trajectories.**
   If one fixed MLP2 dictionary cannot serve both backgrounds, test a small shared
   trunk plus explicitly priced private correction. This is falsifiable against the
   equal-price robust fit.
4. **Independent MLP1 compression and MLP0×MLP1 composition.** This determines
   whether the incompatibility is specific to the MLP2 factorization or a general
   early-interface problem.
5. **Resolve the four-copy-head bundle into selective interactions.** Use subsets or
   a structured Shapley screen with off-target controls to seek the first terminal
   extraction/removal action.

Priority 1 is the highest-information safe next experiment. A GPU job is currently
active in the shared workspace, so the useful immediate work is to preregister and
implement its CPU-side two-background capture/loss contract without opening new
evaluation rows.

That CPU-side action is now complete: the prospective protocol is frozen in
`MLP2_TRAJECTORY_ROBUST_R512_V1_PREREGISTRATION.md`, and the balanced normalized
two-background loss plus minimax checkpoint rule have a focused known-answer test
suite. No new row, model, or outcome was opened. This is an implemented prerequisite,
not a scientific outcome.

## Evidence

- Result: `mlp0_c512_mlp2_full512_composition_v2_result.json`
- Receipt: `mlp0_c512_mlp2_full512_composition_v2_receipt.json`
- Result SHA256: `97822bd27e9b4fca0768da16e7df3c5adbacc2fedcaa6652a715394a5f380f1b`
- Receipt status: `result_complete_receipt_last`
- Runtime: `20.293899536132812` seconds
