# A concrete second-stage result: share input readers across quartic products

22 September 2026, 02:38 UTC. **We can remove128 separate input projections from each of the two CP approximations while closely preserving their outputs and changes across contexts.** This saves about 6% of stored numerical arrays. It is a useful graph simplification of the fitted programs, not a newly identified semantic circuit or a repair of their remaining model-reconstruction errors.

## Where this sits in the model

The underlying target remains the pure quartic contribution obtained by feeding MLP16's quadratic contribution into both inputs of MLP17, in the 18-block model. We compare 16 fixed output coordinates of that contribution. Other residual, attention, bias and cross terms are outside this target; actual normalization and final softcapping remain separate model operations.

The CP approximation expresses each output as a sum of 512 terms. Each term is a product of four learned linear projections of the 1152-dimensional input. Before this edit, the program computes 4×512=2048 linear projections separately, then uses1536 variable multiplications to form the quartic terms. A “reader” here means one of those linear projections, not an attention head, token, or semantic concept.

The decomposition stage proposed these products. This second stage changes how their inputs are computed: several products can now consume the same previously computed reader.

```mermaid
flowchart LR
    X[1152-dimensional normalized input] --> B[Bank of 1920 distinct linear readers]
    B --> P[512 quartic products reuse indexed readers]
    P --> Y[16 fixed output coordinates]
```

The original programs had no exact repeated signed readers. The successful edit therefore uses approximate reader substitution, followed by an exact compiler that caches the resulting repeated readers. These two steps have different correctness claims.

## How a substitution is proposed and checked

Suppose a quartic term uses four scalar features a,b,c,d. We propose replacing its reader a by another reader q already used somewhere in the graph:

$$
abcd\quad\longrightarrow\quad\beta qbcd.
$$

Candidate q comes from similarity under the calibration input second moment. Similarity alone is insufficient: the other three factors can amplify a small reader error. We therefore choose beta to minimize the complete term's expected squared change under the calibration Gaussian:

$$
\beta=\frac{\mathbb E[a q b^2c^2d^2]}{\mathbb E[q^2b^2c^2d^2]}.
$$

These expectations are calculated analytically from weights and the input mean/covariance. We fold beta into the term's output coefficient. Proposals are ranked by their expected output error, without native evaluation labels. We limit substitutions to one source reader per affected atom and never remove a reader retained as another substitution's target.

We registered 32/64/128 substitutions, with 128 as the primary comparison. The new program retains 512 quartic terms; it saves input projections rather than deleting terms. Its polynomial remains homogeneous degree four and even in its input, unlike the earlier nonhomogeneous Gaussian conditional approximations.

## How much changes?

Errors below compare the edited program with its CP parent, not with the native model.

| Primary128-reader edit | Seed 1001 | Seed 1002 |
| --- | ---: | ---: |
| Original2048-state panel: value deviation | 0.245% | 0.170% |
| Original panel: finite-response deviation | 0.474% | 0.354% |
| Larger16384-state panel: value deviation | 0.219% | 0.219% |
| Larger2494-pair diagnostic: response deviation | 0.327% | 0.420% |
| Largest per-output value deviation on larger panel | 1.23% | 1.20% |
| Largest per-output response deviation on larger panel | 1.10% | 1.28% |

Both pass the registered pooled1% value-and-response preservation screen. Some individual coordinates exceed1%, so the result is not uniform sub-1% fidelity. The larger panel was already inspected in earlier work and is an opened diagnostic, not fresh validation.

The compiled float32 exports agree with their float64 edited programs to about 1.8e-7 relative error. Independent controls cover shared readers, squared features, fourth powers, signs and cross-slot permutations. Gaussian quadrature independently verifies the replacement scale and term-error calculation.

## Is sharing better than simply deleting weak terms?

For a direct control, we deleted the 128 quartic terms whose readers were substituted, without refitting anything. Deletion is cheaper still, but changes parent values by 4.82% /2.17% and finite responses by 7.21% /4.26% on the larger panel. Reader sharing changes those responses by only 0.33% /0.42%. The useful computation in these terms should therefore not simply be discarded at this fidelity target.

This control is not an exhaustive pruning search or a matched-cost optimization. It answers the narrower question of whether the very terms touched by the successful edit are individually dispensable.

## What is actually saved?

| Cost | Original CP program | Shared-reader program |
| --- | ---: | ---: |
| Distinct1152-dimensional linear readers | 2048 | 1920 |
| Variable multiplications for quartic terms | 1536 | 1536 |
| Stored floating coefficients, including common writer | 2,385,920 | 2,238,464 |
| Additional reader-address indices | 0 | 2048 int32 |
| Numerical-array storage at float32/int32 | 9,543,680 bytes | 8,962,048 bytes |

The array reduction is about 6.1%; file metadata overhead is separate. We have not benchmarked this new program's runtime. One larger matrix multiplication, indexed reads and memory traffic can affect speed differently from scalar operation counts.

## What this does and does not establish

This demonstrates the intended decomposition-to-graph workflow: use learned product features as candidates, introduce reuse, price the whole program and test the effect. Both primary edited programs are exported and reproducible.

Native reconstruction remains roughly7.39% /7.57% pooled error on the larger panel, with much larger errors in small features. The CP parents' earlier native removal test failed its absolute and improvement criteria; preserving those parents is not equivalent to recovering valid causal circuits. Selective semantics, fresh/OOD transfer and composition are still unproven. The separate queued residual-learning experiment aims to improve the missing computations before further simplification.

[Registered edit protocol](../../direct_tensor_match/CP_LINEAR_REUSE_PLAN_V1.md) · [All native-panel comparisons and proposals](../../direct_tensor_match/CP_LINEAR_REUSE_V1.json) · [Export checks and per-output errors](../../direct_tensor_match/CP_LINEAR_REUSE_EXPORT_V1.json) · [Deletion control](../../direct_tensor_match/CP_LINEAR_REUSE_DELETION_CONTROL_V1.json) · [Compiler](../../direct_tensor_match/cp_linear_reuse.py).

## 02:51 a complementary structural baseline for the single layer

A separate check returned to the native last MLP's third-order quadratic tensor, rather than the quartic approximations. Could all selected outputs use one shared orthogonal set of input features and only squares of those features? The native quadratic matrices do not commute, ruling out that exact restricted representation. For eight fixed, separately normalized output pairs, an analytic commutator inequality gives numerical coefficient-error lower bounds of **5.4–7.0%**.

This does not rule out general Tucker: a full Tucker core can retain interactions between different input features. It also leaves nonorthogonal dictionaries, block terms and bilinear DAGs available. The result separates one structural restriction from optimizer failure; it is not a lower bound on text prediction error. [Derivation, controls and precise scope](../../direct_tensor_match/JOINT_SQUARE_BASIS_INTERPRETATION_V1.md).


## 03:50 follow-up: shared readers expose shared quadratic products

The second-stage graph search now finds an additional exact saving. After the reader-sharing edit, reassociating the quartic products reduces nonlinear multiplication counts from the naive1,536to**1,527and1,525**. One product in the second candidate was already shared by the old fixed tree, so the additional savings are9and10.

For example, instead of computing $(ac)(bd)$ and $(ae)(bf)$ separately, the graph can compute $q=ab$ once and reuse it in $q(cd)$ and $q(ef)$. The compiler also permits cubic intermediates, with an exact planted control, although these two native candidate graphs use quadratic sharing only.

```mermaid
flowchart LR
  A[Frozen quartic CP candidate] --> B[Approximate linear-reader sharing]
  B --> C[Exact product reassociation]
  C --> D[Cache repeated quadratic products]
  D --> E[Same candidate function with fewer products]
```

Both graphs were exported and reloaded; float64 replay errors are below $7.5\times10^{-16}$. Dense input projections and output coefficients are unchanged, so this modest operation-count saving is not a demonstrated runtime improvement. The rewrite also preserves the candidate's existing native-fidelity failures. It is concrete progress on graph simplification, not a newly identified circuit. [Exact controls, accounting, and graph exports](../../direct_tensor_match/QUARTIC_PRODUCT_REASSOCIATION_INTERPRETATION_V1.md).

## 03:54 precision check: finite changes survive the exported graph

The exported graphs also preserve their source candidates in float32 on all 16,384 opened states and 2,494 matched pairs. The largest per-output relative response difference is **1.27e-6**; both candidates pass all registered pooled and per-output numerical thresholds. This is a CPU transformation-preservation check, not new native accuracy or OOD evidence. It leaves the source candidates' much larger component-fidelity failures unchanged. [All coordinate results and precise comparison](../../direct_tensor_match/PRODUCT_DAG_RESPONSE_REPLAY_INTERPRETATION_V1.md).

## 03:59 follow-up: approximate sharing of whole quadratic products

A more aggressive edit replaces one quadratic intermediate with another existing quadratic, using a weight-derived scale. The registered 128-edit version fails: parent response deviations are 3.45–4.33%, above the 1% limit, and the worst individual outputs change by 8.12–11.60% in responses.

At 64 edits, the program retains 1,920 linear readers—the same number as the earlier linear-sharing candidate—but uses 1,472 nonlinear products instead of 1,527/1,525. That saving comes with worse preservation of some individual outputs. This smaller setting is a descriptive tradeoff, not a replacement for the failed primary test. Native small-output response errors still remain near 58–60%, and no new candidate was exported. [Detailed accounting, controls and both fidelity comparisons](../../direct_tensor_match/CP_QUADRATIC_REUSE_INTERPRETATION_V1.md).
