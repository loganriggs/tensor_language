# Stronger baselines at the wider graph's storage budget

21 September 2026, 10:20 UTC. **The wider shared graph needs to beat a stronger comparison than the old 768-product baseline.** Independent pair programs at almost identical coefficient storage reach 8.81% third-component error on the opened states. We also counted all scalar multiplications: fewer activation-by-activation products do not imply less total arithmetic when dense projections dominate.

The registered 12-fit graph comparison is still running. These controls were constructed independently of its final outcomes, and do not change its original registered verdicts.

**Executable baselines**

Each component gets its own input subspace and two quadratic forms. We choose the subspace from that pair's mode Gram matrix, then use the existing symmetric-pair compiler to express both forms with shared products. This is an exact compilation within the chosen approximate subspace; it is not a globally optimal decomposition.

All cases retain the same original component definitions, affine/mean corrections, explicit normalization, and supplied native inputs. The output writer is stored once across all three components.

| Baseline geometry / width per pair | Products | Floating coefficients | Component errors |
|---|---:|---:|---|
| Native isotropic / 256 | 768 | 897,804 | 2.31%, 3.53%, 16.27% |
| Calibration-shaped / 256 | 768 | 897,804 | 3.06%, 2.76%, 11.94% |
| Native isotropic / 384 | 1,152 | 1,340,940 | 1.62%, 3.06%, 11.26% |
| Calibration-shaped / 384 | 1,152 | 1,340,940 | 1.87%, 2.05%, 8.81% |

The pending wider shared graph has **592 products and 1,342,028 floating coefficients**, only 1,088 more coefficients than these wider baselines. It may use substantially fewer nonlinear products, but its accuracy must be compared at this capacity too.

At width 384, native-isotropic fitting gives 46.44% native coefficient error and 12.26% calibration-shaped error. Calibration-shaped fitting gives 55.26% native and 7.03% calibration-shaped error. Neither metric should be silently substituted for the other.

The old covariance-shaped width-256 component errors replay within $10^{-7}$. All four compilations succeed. Independent native-form execution and the existing residual-write executor agree below $10^{-8}$. Construction and checks took 8.46 seconds on two CPU threads; no model forward or additional GPU process was used.

**What the product savings actually mean**

There are two kinds of multiplication here: multiplying two activations, and multiplying an activation by a learned coefficient inside a dense linear projection or readout. Earlier “product counts” refer to the former, which counts nonlinear interaction nodes. The latter still dominates conventional scalar operation counts.

| Source-stage operation count | Separate 384-per-pair baseline | Shared 560+32 graph |
|---|---:|---:|
| Input-projection coefficient multiplications | 1,327,104 | 1,327,104 |
| Output-mixture coefficient multiplications | 2,304 | 3,392 |
| Activation-by-activation products | 1,152 | 592 |
| Total scalar multiplications | **1,330,560** | **1,331,088** |

Thus the shared graph saves 560 nonlinear product nodes but has **528 more scalar multiplications overall** in this source stage. These totals exclude the same affine and later-component operations in both programs. The pair compiler also uses additions/subtractions within two-dimensional blocks; these are included separately in the detailed receipt. Pair-index storage is 3,456 integers for the separate program, versus one scalar output index for the hard-coded mixed/private graph layout.

This does not make product sharing meaningless: fewer distinct interactions can be a useful form of circuit simplicity. It does mean that we have not demonstrated a runtime advantage. A convincing reusable program must be judged on its full arithmetic, storage, behavior and interpretability rather than a single count.

**How these controls will be used**

The pending fit's preregistered limits stay intact. Its results will also be displayed against both same-storage pair baselines. Passing an older comparison will not be reported as dominating these stronger controls. All current errors are on repeatedly inspected diagnostic states; a frozen candidate would still need fresh native intervention tests, stable identification and explicit treatment of its native-input dependencies.

Evidence: [baseline programs and checks](../../direct_tensor_match/COST_MATCHED_PAIR_BASELINES_V1.json), [literal arithmetic accounting](../../direct_tensor_match/PAIR_GRAPH_ARITHMETIC_PRICE_V1.json), and [registered shared-graph comparison](../../direct_tensor_match/DUAL_GEOMETRY_SOURCE_PLAN_V1.json).
