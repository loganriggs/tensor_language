# Best current circuit: the aligned v23 four-head `is`/`was` program

**Status:** identified partial circuit; not yet a standalone whole-model replacement
**Model:** `bilin18` (18 transformer blocks, residual width 1,152, nine 128-dimensional
attention heads per block, and 4,608 bilinear MLP factors per block)
**Circuit:** `L8H1 + L9H1 + L9H4 + L11H3`

## High-level explanation

The best-understood current circuit changes whether the model prefers `is` or `was`. Four attention
heads write a distributed causal signal:

```text
L8H1 ---- transformed through blocks 9 and 10 ---\
L9H1 -------- transformed through block 10 -------+--> block-11 residual --> later decoder --> is/was margin
L9H4 ---------------- earlier aligned route -------+
L11H3 -- exact live-RMSNorm transport -------------/
                                                     \
                                                      --> small MLP11 branch
```

On fresh, position-aligned v23 examples, installing the four donor head-value slices in a base run
reproduces `72.6%` of the target behavioral change with cosine `.995`. Removing the same four heads
erases `72.2%` of the native change with cosine `.994`. The intervention is stable across reporter
halves, all four heads receive positive Shapley credit, and leakage is small on the strongest
aligned control. This combination of sufficiency, necessity, selectivity, exact transport, and
literal-weight analysis is why this is the best *explained* circuit, even though another H4 circuit
currently has broader cross-task composition evidence.

The important structural result is that the four heads are neither one head in disguise nor four
copies of an identical matrix. They share a strong low-dimensional physical context-reading core,
then use head-private adapters and tails. MLP11 reads a real but minor part of the signal; most of
the four-head behavioral effect continues through the residual stream. The exact later consumers
of that dominant branch are the main unresolved part of the circuit.

## 1. Task, corpus, and counterfactual

The task uses controlled sentence pairs whose desired answer differs along the copular tense
contrast `is` versus `was`. Each row has:

- a **base** sequence, which produces the computation being edited;
- a **donor** sequence, which supplies the counterfactual internal state;
- a correct answer token and a foil token;
- a declared semantic endpoint at which the same computation is compared.

V23 contains 64 length- and endpoint-aligned rows: 16 each in target families `A1` and `A2`, a hard
answer-preserving control family `P`, and a canonical aligned control family `C`. Thirty target rows
are jointly capable: the unmodified model gives the required base and donor behavior, so a causal
effect is well-defined. Selection of the four heads was frozen before v23 outcomes were opened;
v23 did not tune a new rank, head set, or intervention dose.

The scalar behavior is the answer margin

```text
m(x) = logit(correct | x) - logit(foil | x).
```

For a base `b`, donor `d`, and intervention `p`, rowwise recovery is

```text
r = (m_p - m_b) / (m_d - m_b).
```

The reported circuit-level response is evaluated as a vector across rows. If `delta_p` is the
patched response and `delta_t` is the full target response, then

```text
signed projection = <delta_p, delta_t> / ||delta_t||^2
cosine           = <delta_p, delta_t> / (||delta_p|| ||delta_t||)
relative residual = ||delta_p - delta_t|| / ||delta_t||.
```

Signed projection measures recovered scale; cosine measures directional agreement; residual
measures all remaining vector error. Reporting only one would be misleading.

## 2. What is patched

`LlHh` denotes attention head `h` in transformer block `l`, with zero-based indices. Let `z_h^b`
and `z_h^d` be a selected head's complete 128-dimensional slice at the input to attention's output
projection (`c_proj`) on matched base and donor runs. At every token through the registered semantic
endpoint, the four-head installation replaces the corresponding base slice by its donor value:

```text
z_h^patch = z_h^b + (z_h^d - z_h^b),  h in {L8H1,L9H1,L9H4,L11H3}.
```

All other native computation remains live. This is a full-head intervention, not a fitted DAS
coordinate. The reverse-removal experiment applies the opposite causal contrast to test necessity.
Self-patches must be exactly inert, and complete donor patches must reproduce the donor hidden state
and logits; all three closure checks are zero in the confirmation receipt.

Two response spaces are scored:

- **Behavior:** the vector of answer-margin changes.
- **Reader-contracted `Q`:** the induced MLP11 hidden-factor changes after contraction with a frozen
  downstream task reader. `Q` tests whether the same intervention arrives in the task-relevant
  MLP11 coordinates, not merely somewhere in its large hidden state.

## 3. Prospective confirmation result

| Measurement | Behavior | MLP11 reader-contracted `Q` |
|---|---:|---:|
| Four-head target signed projection | `.72588` | `.67007` |
| Target cosine | `.99524` | `.97676` |
| Correct direction fraction | `1.000` | `1.000` |
| Hard-control `P` leakage | `.10033` | `.08344` |
| Canonical-control `C` leakage | `.00396` | `.00640` |
| First reporter half | `.72045` | `.66044` |
| Second reporter half | `.73085` | `.67637` |

Every frozen singleton screen passed. All 64 rows were constructed before model access, and the
confirmation used 19 model forwards, 1,216 sequence evaluations, and one backward pass. It fit no
intervention parameters.

The hard P control is deliberately much more difficult than C because it shares nuisance temporal
and lexical structure while preserving the answer. Its roughly `.10` leakage is not zero, so the
circuit is selective rather than perfectly exclusive. The near-zero C value rules out the simpler
explanation that any aligned donor head replacement produces the effect.

## 4. Distributed credit rather than a single magic head

For the finite game `f(S)` over subsets of the four heads, the Möbius/Harsanyi dividend is

```text
mu(S) = sum_{T subseteq S} (-1)^(|S|-|T|) f(T).
```

Shapley credit gives each head its singleton effect plus an equal share of every interaction in
which it participates. The behavior allocation is:

| Head | Behavior Shapley credit | `Q` Shapley credit |
|---|---:|---:|
| `L9H4` | `.22670` | `.26322` |
| `L9H1` | `.21735` | `.13014` |
| `L8H1` | `.14838` | `.14190` |
| `L11H3` | `.13346` | `.13480` |

The Shapley efficiency residual is at numerical zero (`<= 1.1e-16`), and pair interactions are
small compared with the union. Thus the effect is distributed but close to additive under this
intervention. Shapley is an accounting rule, not a directed-edge proof; the physical transport and
loss/rescue experiments below supply that evidence.

## 5. Necessity and the MLP11 branch

Reverse removal of the four heads has behavior signed projection `.72151`, cosine `.99401`, and
correct direction on every target row. It is stable across halves (`.72567` and `.71771`), while P
and C leakage are `.11861` and `.00416`. The forward-installation and reverse-removal magnitudes are
therefore strikingly symmetric.

MLP11 is not the dominant mediator:

- restoring the complete induced MLP11-factor change rescues only `.15114` of the removed-head
  behavior;
- the leading occupied MLP11 mode explains `.87376` of that *small local rescue*;
- direct removal of that mode accounts for `.18197` of the full four-head effect.

The correct interpretation is “necessary four-head circuit with a minor MLP11 branch.” It is not
“one MLP11 mode explains the four-head mechanism.” A pending exact residual-by-MLP11 factorial is
designed to close the decomposition and identify how much is separately rescued by each branch.

## 6. The four heads take different transport paths

Layerwise interventions locate the earliest boundary at which each head's eventual response becomes
well aligned with the target:

- `L9H4` is already strongly aligned at block-10 input, after block 9: projection `.84878`, cosine
  `.97824`.
- `L9H1` is poorly oriented at that same boundary (projection `.6666`, cosine `.6729`), becomes
  aligned by block-11 input after block 10 (`.70475`, `.98335`), and still needs block-11 attention
  for exact closure.
- `L8H1` rotates progressively: block-9 input (`.62036`, `.7440`), block-10 input (`.73958`,
  `.94337`), block-11 input (`.68739`, `.98464`), then exact closure after block-11 attention.
- `L11H3`, being adjacent to the measured boundary, exposes the live normalization step directly.

These differing trajectories are evidence against replacing the circuit by one arbitrary pooled
activation direction. The same final causal variable can be written at different times and
transported through different native computations.

## 7. Exact live-RMSNorm transport for `L11H3`

Let RMS normalization be

```text
N(x) = g odot x / s,
s = sqrt((x dot x)/d + epsilon),
```

where `g` is the learned gain and `d=1152`. Its Jacobian applied to a perturbation `v` is

```text
J_N(x) v = g odot [v/s - x (x dot v)/(d s^3)].
```

The exact finite intervention recomputes `N(x + delta)` rather than pretending normalization is a
fixed linear matrix. Exact projected transport closes with relative error `9.26e-7`, and both native
and patched inputs reconstruct within `1.91e-6`. The first-order tangent approximation has cosine
`.999984`, signed projection `.992374`, and residual `.009452`. The remaining finite nonlinear term
has target RMS `.0390`, versus `.00650` on P and `.000400` on C.

This is why a direct static `W_O -> MLP11` contraction originally failed: the physical computation
between those weights includes input-dependent RMS normalization. Once that operation is included,
the local transport becomes essentially exact.

## 8. Exact bilinear reader computation

For MLP11 residual input `x`, hidden factor `n` is

```text
H_n(x) = (l_n^T x)(r_n^T x),
```

and the MLP output is

```text
y = sum_n d_n H_n(x),
```

where `l_n`, `r_n`, and `d_n` are rows/columns of the checkpoint's left, right, and down matrices.
Given a frozen downstream behavioral gradient `g`, define the factor reader

```text
rho_n = g^T d_n.
```

The reader-contracted scalar response is

```text
Q = sum_n rho_n Delta H_n.
```

For a head write `Wz`, the exact finite hidden-factor substitution is

```text
Delta H_n = (l_n^T x)(r_n^T Wz)
          + (r_n^T x)(l_n^T Wz)
          + (l_n^T Wz)(r_n^T Wz).
```

The first two terms are context-by-write interactions; the third is a write-by-write term. This
formula connects a full-head causal intervention to literal checkpoint weights without fitting a
post-hoc activation axis.

Equivalently, the complete bilinear MLP tensor is

```text
T[o,i,j] = sum_n Down[o,n] Left[n,i] Right[n,j],
```

so that `y_o = sum_ij T[o,i,j] x_i x_j`. Restricting `i`, `j`, or `o` to a causally established
interface produces an exact smaller tensor for that interface.

## 9. Capability in weights versus occupancy on data

The exact MLP11-restricted weight tensor has broad *capability*: many factors could legally respond
inside the interface. Native factor-score patterns are very similar across heads (median cosine
`.98538`), yet the top 256 of 4,608 factors hold only `.28090` of the total mass. In contrast,
`.97137` of the realized v23 target response occupies one frozen reader mode, stable across reporter
halves at cosine `.99998`.

This separates two questions:

1. **Weight capability:** what computations can the checkpoint perform in the admitted subspace?
2. **Activation occupancy:** which of those computations does this corpus actually visit?

PCA, SAE, hierarchical SAE, or clustering belongs to the second question. Such a model may describe
sparse or hierarchical use of the circuit, but it cannot by itself define the causal circuit,
because unobserved legal computations disappear and frequent irrelevant states can dominate.

## 10. Shared core plus private adapters

After contracting the occupied reader mode, each head has a physical map `M_h` from shared residual
context into its private 128-dimensional head coordinates. Raw maps have low median cosine
(`.04860`), while their leading context directions overlap strongly (`.939-.982`) and their
gauge-invariant context-Gram similarity has median `.92455`. Full-map Procrustes similarity is only
`.68596`; top-8 context-projector overlap is `.39605`, and private-coordinate overlap is not
meaningful without fixing each head's gauge.

The current weight hypothesis is therefore

```text
M_h = U A_h + E_h,
```

where `U` is a shared physical context basis, `A_h` is a head-private adapter, and `E_h` is a private
tail. For a fixed rank `k` and explicit component weights `w_h`, the globally optimal common basis
solves

```text
minimize over U^T U = I:  sum_h w_h ||M_h - U U^T M_h||_F^2.
```

It is obtained from the first `k` left singular vectors of

```text
X = [sqrt(w_1) M_1 | ... | sqrt(w_H) M_H].
```

The sum of discarded squared singular values is the exact optimum certificate. Leave-one-head-out
fits test whether the core predicts an omitted physical writer rather than merely fitting a pooled
average. The first rank-one receipt is intentionally quarantined because its float32 certificate
missed the preregistered absolute tolerance; its unchanged scientific test is being repaired in
float64. No semantic rank is claimed from the opened singular-value ladder.

## 11. Evidence scorecard

| Circuit requirement | Current status | Evidence or gap |
|---|---|---|
| Computational specification | Partial pass | Four writers, per-head transport, exact RMSNorm and MLP11 tensor; dominant later reader unresolved |
| Cross-module grouping/splitting | Partial pass | Four-head group and shared-core/private-tail hypothesis; causal core/tail swaps pending |
| Held-out/OOD prediction | Pass on aligned v23 | Frozen heads, all families aligned before access; completely new v24 constructions pending |
| Sufficiency/extraction | Partial pass | `.72588` causal installation and exact local tensors; not a standalone full-effect executor |
| Selective manipulation | Pass for current claim | Reverse removal `.72151`, low C leakage, modest P leakage; downstream branch edits pending |
| Composition/reuse | Not yet for this exact circuit | The related H4 circuit passes cross-task composition; v23 shared-core composition remains untested |
| Stable identification | Substantial pass | Reporter halves, exact closures, full-head interventions, gauge audit; broader corpus and restart tests remain |
| Simplicity/adoption | Not reached | No complete literal-cost comparison or whole-model replacement |

This scorecard is why the object is called an **identified partial circuit**, not a solved model.

## 12. What would falsify or upgrade the account

The next decisive tests are already specified:

1. **Residual/MLP11 factorial:** exact residual-only, MLP11-only, and joint restoration must close.
   Failure of joint closure would invalidate the branch accounting instrument.
2. **Reciprocal downstream atlas:** each complete module from A12/M12 through A17/M17 is tested by
   transfer sufficiency and reset necessity; any passing attention module is then split into heads.
3. **Unfiltered v24 transfer:** new markers, reporters, constructions, and every target row are used
   without correctness filtering or reranking. Failure would scope v23 to its construction family.
4. **Common-core versus private-tail swaps:** prospectively fixed `UU^T M_h` and
   `(I-UU^T)M_h` interventions must reconstruct the parent effect and beat matched-energy rotated
   controls. Otherwise the shared-looking weight geometry is not a reusable circuit variable.
5. **Composition and extraction:** the fixed shared core must predict joint behavior with another
   task circuit, after which an executable reduced tensor program and literal storage/compute price
   can be evaluated.

## 13. Main evidence artifacts

- Confirmation: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_aligned_four_head_reader_contracted_confirmation_v1_result.json`
- Necessity and MLP11 rescue: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_four_head_necessity_occupied_mode_rescue_v1_result.json`
- Exact RMSNorm transport: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_l11h3_exact_rmsnorm_transport_v1_result.json`
- L9 transport: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_l9h1h4_layerwise_transport_localization_v1_result.json`
- L8 transport: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_l8h1_layerwise_transport_localization_v1_result.json`
- Restricted weight tensor: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_four_head_m11_restricted_weight_capability_tensor_v1_result.json`
- Writer grouping: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_grouping_v1_result.json`
- Gauge audit: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v23_occupied_reader_weight_writer_gauge_audit_v1_result.json`
- Broader canonical dossier: `basis_aligned/polynomial_causal/explanations/CIRCUIT_temporal_iswas_rank46_task_modes_2026-09-07.md`
