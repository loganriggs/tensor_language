# Morning synthesis — six independent routes to a compiled bilin18

## Goal and standard of success

The goal is not merely to approximate weights or lower validation loss.  It is to replace the checkpoint by a
substantially smaller **tensor program** whose predictions compose across independently compiled parts, whose
named interventions still transport with the right sign and relative magnitude, and whose complete executable
state has a literal scalar and byte bill.  A useful explanation should therefore answer four questions at once:

1. **Predictive:** does it preserve held-out and shifted-corpus cross entropy?
2. **Composable:** do separately fit replacements still work when installed together?
3. **Manipulable:** do circuit ablations retain their signed effects, rather than merely their average loss?
4. **Simple in reality:** how many scalars and bytes must be stored, including routers, indices, and coefficients?

Generic input-dependent top-k is intentionally outside the structural claim.  Its possible active sets grow
combinatorially, so it is a compute policy unless the selector itself is represented and priced.  A finite MoE
router with a small, fixed state set and a fixed expert subset per state is a valid tensor program.

## Outcome in one paragraph

The strongest overnight result came from combining route 1's bilinear viewpoint with route 5's calibrated error
contracts.  The ordinary Frobenius geometry was wrong: under the live input covariance, **all 18 MLP input maps**
admit accurate shared-input rank reductions.  A preregistered rule selected two mild p768 cuts at layers 4 and 0
and combined them with context-metric Q/K rank64.  After source-aware BF16 storage and FP16 Q/K factors, the fully
gated artifact has **511,758,646 semantic scalars and 1,023,517,292 stored bytes**, versus 545,902,902 scalars and
2,067,669,612 bytes natively.  It adds `.012329` census CE, retains 43/62 certificates, transfers to a disjoint
WikiText population, and matches the original-native signed a16 intervention with cosine `.986524` and collateral
Spearman `.995306`.  This is a 6.2546% semantic-scalar reduction and 50.4990% byte reduction.  Compute remains fp32
after dequantization, so no runtime or activation-memory claim is being made.

## Common scorecard

Scores are evidence-weighted, not estimates of an untested best case.  Compute cost is the cost of the next
decisive test, not the cost already spent.

| Rank | Independent route | Measured signal | Compression actually earned | Identifiability | Causal relevance | Next-test cost | Verdict / decisive next experiment |
|---:|---|---|---|---|---|---|---|
| 1 | Executable error contracts and lower bounds | Family tail laws have log-space R2 `.995`; the Q/K certificate-damage ray has R2 `.99945`; fixed-ray transfer to three distributed-MLP programs has cosine `.9988–.9994` and count error `1–2` | No parameters by itself, but it selected and certified the adopted 34.14M-scalar reduction | High across every measured Q/K, MLP0, value, and mid-stack-MLP program; still empirical outside them | High: predicts certificate survival, not only CE | Low CPU calculation, then one GPU confirmation | **Use as standing allocator.** Require ray headroom before any new representation reaches a physical build; do not spend GPU on a candidate already beyond the desired certificate transition |
| 2 | Direct bilinear-MLP compression | Native CP-atom sharing and a low-rank layer axis failed; context-metric shared-input rank succeeded at all 18 layers; two selected layers compose cleanly | 5,308,416 scalars removed by the adopted two p768 MLP cuts | High for installed subspaces; all-layer tail transfer factor `1.223x`/rho `.952`; low for raw CP atoms | High after the installed pair passed signed transport | High for a truly new factorization; low-value local grid is closed | **Keep adopted point; close local tuning.** Exact variable-rank water-filling reselects `{0,4}@p768`, while its next cheaper unit predicts `1.369x` damage |
| 3 | Joint untied-vocabulary factorization | Frequency-weighted shared code scores `+.193/.225` FW/Wiki versus matched independent `+.552/.778`; later r768 improves to `+.127/.145` but unseen targets remain `+.724/.566` | 30.28M vocabulary scalars at the first screen, but no legal point earned adoption | Medium-high shared-geometry signal; rare error is distributed, not a sparse exception table | None | High relative to current damage | **Stop adoption route.** Sparse Fisher/norm residuals repaired only 1.8–4.4%, and the distributed-rank frontier still failed absolute predictive bars |
| 4 | Exact embedding-folded MLP0 structural recovery | Planted blocks are recoverable under the right prior; real contraction algebra is irreducible under tested block/tree nulls; token and context finite routers lose badly to one shared subspace | No structural saving earned; contextual p512/p768 spectral maps do earn ordinary low-rank savings | Strong negative result: raw factors are gauge non-identifiable; invariant contraction algebra is the proper object | Potentially high, but a DAG is not orientable without asymmetric interventions | Low for a named conditioned algebra; high for blind structure search | **Close generic block/tree/DAG search.** Reopen only with an independently named behavior/intervention that defines conditioned contractions and supported finite states |
| 5 | Causal-response coordinates for rank allocation | Signed-response r128 is usable (`+.0875/.0616`) but loses to activation PCA (`+.0500/.0302`) and weight SVD; split overlap `.283` | 4.57M MLP0 scalars at r128, but no advantage over noncausal baselines | Low-medium: the empirical suffix gradient is noisy and split-unstable | Local and nominally causal, but not an actual intervention basis | Low | **Kill the scalar response eigenbasis.** A successor must preserve a vector-valued suffix-Jacobian or several named interventions, not reweight one covariance by another scalar |
| 6 | Predictive causal-state quotient / Hankel structure | Quote state is classifiable and transfers at `.875`, but state R2 fails a live shuffle control; head13.8 explains only 6–7% of separation | No priced native replacement | Low for a generative quotient; high only for the weak classifier claim | Circuit-local, redundant, and suffix-bank dependent | Low, but low upside | **Kill as compiler route.** Keep quote state as a circuit classifier; reopen finite state only when behavior supplies a natural supported automaton |

## What the embedding-folded MLP0 idea did and did not establish

For a bilinear MLP written schematically as

`f(x) = D[(Lx) ⊙ (Rx)] + b`,

the basis-invariant object is the symmetric third-order contraction

`T_j = sum_u D[j,u] sym(L[u,:] ⊗ R[u,:])`.

At token position zero, every possible input is known: `x=E[t]` for each vocabulary item.  Thus the entire
embedding-fed function can be folded exactly into `T(E[t],E[t])` and exhaustively evaluated over the finite
vocabulary.  This removes activation-sampling uncertainty, but it does **not** remove gauge ambiguity.  Different
bilinear factors can implement the same `T`, so raw neuron supports cannot identify blocks, a hierarchy, or a DAG.

The invariant tests are sharper:

- Blocks require nontrivial idempotents in the commutant of the contraction slices.
- A hierarchy requires a nested flag of reducing subspaces.
- A directed acyclic graph requires an asymmetric relation—an ordered intervention, time direction, or distinct
  input roles.  A single commutative quadratic map cannot orient an edge from observational weights alone.

The planted positive controls recover their intended partitions when the prior is supplied, while incompatible
priors fit essentially the same function and the real-model commutant shows no robust reducing decomposition.
That is an identifiability result, not a failure to optimize.  It redirects the useful part of the idea toward
context-covariance spectral compression, where an executable input encoder is directly identified and priced.

## Laws that now replace rank sweeps

Let `C_f` be the live input covariance for component family `f`, and let `T_f(r)` be the omitted singular energy
after rank `r` approximation of `W_f C_f^(1/2)`.  The measured physical damage follows

`D_f(r) = a_f T_f(r)^(b_f)`.

For Q/K and MLP0 the log-space fits are about `.995`, with exponents approximately `1.69` and `1.08`; their gains
differ by `3.30x`.  Equal omitted weight energy is therefore not equal consequence.  The allocator should spend
the next scalar where the **family-calibrated marginal damage per saved scalar** is smallest.

Certificate behavior is nearly one-dimensional.  If `v_i` is the damage to circuit member `i`, the measured
programs obey `v_i ≈ s k_i`.  The Q/K ray fit has R2 `.99945`, transfers to held-out mixed and value programs at
cosine above `.997`, and converts a proposed scalar intensity `s` into a predicted count

`#{i : s k_i < 1}`.

Together these two laws turned a 42-choice Q/K×MLP0 grid into an exhaustive CPU calculation.  It certified that no
point in that grid below 512,561,462 scalars conservatively retains 43 certificates.  The later MLP4 discovery did
not contradict the certificate: it moved outside its stated grid and then won in one preregistered physical build.
The same fixed ray subsequently predicts the selected two-layer, third-layer, and QK72 mid-tier programs with
certificate-count errors `2/1/1`, so it is now a measured cross-family allocator rather than a Q/K-only curiosity.
A preregistered second-mode test finds stable residual curvature and improves a held-out distributed program from
R2 `.99316` to `.99910` with exact count prediction, but its value-family specificity control narrowly fails
(`.5257` cosine versus a frozen `.50` ceiling).  It is therefore mapped as universal curvature, not promoted as an
MLP-specific causal coordinate; the one-ray model remains the conservative allocator.

## Negative results that should remain closed

- Per-token top-k can fit MLP0 substantially better, but its combinatorial active-set table is not a small tensor
  network.  It receives no structural compression credit.
- Four-state token-identity and live-context MoE routers are about 10x and 25x worse, respectively, than one
  equal-price global subspace.  The natural four-state morphology follow-up lacked preregistered split support
  (`356/206` digit examples versus `300/half`) and was stopped before arm scoring; the bar was not lowered.
- Cross-layer native-atom reuse is indistinguishable from an orthogonal-coordinate null.  A Grassmann midpoint
  shared encoder for layers 0 and 4 is functionally tolerable, but their normalized rowspace overlap `.6788` is
  near the random expectation `.6667` and below the frozen `.72` structural bar.
- A single symmetric bilinear map cannot identify a DAG direction.  More optimization does not repair missing
  observational information.

## Recommended program after the overnight window

1. Treat the two-byte 511.76M artifact as the current 43-certificate deliverable and the source-aware BF16 native
   point—545.90M scalars, 1.0918GB, census `+.000009`, 62/62 certificates—as the high-fidelity anchor.
2. Keep the third-layer result as a mapped lower-fidelity point, not a frontier: `{4,0,2}@p768 + QK64` reaches
   509,104,438 scalars and good shifted CE but only 38/62 certificates, below both its frozen >=40 signal bar and
   the 43 needed for promotion.  This closes manual prefix extension after two layers.
3. Keep the all-layer variable-rank result as a local optimality certificate.  Its fit-B error is only `1.223x`
   with Spearman `.952`, yet exact water-filling at the current price chooses the already-adopted `{0,4}@p768`
   pair; the next cheaper unit predicts `1.369x` its damage.  Do not run more manual layer/rank variants.
4. Extend the tail law to value maps on CPU before spending GPU time; the measured value96 point is currently much
   less efficient per scalar than Q/K.
5. Treat vocabulary sharing as a representation result, not a live adoption branch.  Both the sparse rare-row
   repair and a distributed-rank frontier have already failed their prospective predictive bars.
6. Once the semantic frontier stabilizes, measure serialized load, dequantization memory, and fused BF16/FP16
   execution.  Storage savings alone do not imply latency savings.

The broad research lesson is that the model is not revealing a clean symbolic block/DAG decomposition in raw
MLP0 weights.  It is revealing a **metric-dependent spectral program** whose error and causal degradation are
regular enough to allocate mathematically.  That is less visually discrete than a module graph, but it is already
predictive, composable, manipulable, and literally cheaper—and every adopted step is backed by a frozen receipt.
