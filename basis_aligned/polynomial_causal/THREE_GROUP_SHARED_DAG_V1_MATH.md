# An exact shared-intermediate compression of the retained attention predictor

13 September2026, setting2. This works on the actual retained three-contraction program. It does not approximate a broader arbitrary-input MLP tensor and does not add a learned factorization.

Write the native two QK score arrays as \(a,b\), their child changes as \(a_c,b_c\), and their remainder changes as \(a_r,b_r\). All are scalar per causal source position. The additive score summaries and their discrepancies from the actually normalized additive input are

$$
\bar a=a+a_c+a_r,\quad \bar b=b+b_c+b_r,\quad
e_a=a_A-\bar a,\quad e_b=b_A-\bar b.
$$

With128-dimensional value changes \(v_c,v_r\) and \(\bar v=v+v_c+v_r\), the retained predictor is

$$
\sum_s w_{c,s}v_{c,s}+\sum_s w_{r,s}v_{r,s}
+\sum_s(e_{a,s}\bar b_s+\bar a_se_{b,s})\bar v_s.
$$

The previous implementation computes

$$
w_c=a_r(b+b_c)+(a+a_c)b_r+a_rb_r,
\quad
w_r=a_c(b+b_r)+(a+a_r)b_c+a_cb_c.
$$

Each uses three scalar products. Distributivity gives

$$
w_c=a_r\bar b+(a+a_c)b_r,
\qquad w_r=a_c\bar b+(a+a_r)b_c.
$$

Each now uses two products, and both reuse the already computed \(\bar b\), which also feeds the inherited term. Including that term, the scalar gate-product count falls from8 to6. This is a concrete shared arithmetic DAG improvement within the retained program; it is not evidence of a shared semantic circuit across tasks.

## Smaller-looking alternative and numerical countercheck

Alternatively form \(P=\bar a\bar b\) once and use

$$
w_c=P-(a+a_c)(b+b_c),\qquad
w_r=P-(a+a_r)(b+b_r).
$$

This reduces the total gate-product count to5, but subtracts potentially close large products. On synthetic quantized FP32 inputs at edit strength1e-4, relative error versus the FP64 reference is6.26e-5, compared with8.77e-7 for both the original and six-product formulas. The test retains all strengths and methods; the five-product variant is not preferred merely because its graph has one fewer multiplication.

## Actual native-port comparison and price

All120 cached native FP64 port contexts give exactly identical residual-write tensors for the original, six-product and five-product implementations. All120 writes also agree after FP32 conversion. These are real cached ports used by the existing predictor, not fresh behavioral examples. The frozen original package and its binding are unchanged; `three_group_shared_dag_v1.py` is a separate candidate.

Seven repeated CPU eager timings of120 core calls give medians6.50ms original,5.89ms six-product and6.04ms five-product. This small microbenchmark includes Python/tensor-dispatch costs; it is not a GPU or whole-model speedup claim.

Each of the three value contractions still needs128 scalar multiplications per source. Counting those, the core goes from392 to390 multiplications per source: **about0.51%**, not25% overall. Input projection, normalization, routing, output projection and downstream MLP costs are unchanged. Stored weight count does not improve. This is a small validated computational simplification, not the large circuit compression sought by the campaign.

The useful lesson is that producer-aware algebra can find real reuse that arbitrary-port spectral tests miss. Larger savings will need reused higher-cost contractions/readers or learned shared blocks, not just polishing this scalar expression. The six-product variant is the candidate to carry into future retained-program comparisons; its existing approximation error relative to full attention remains unchanged.

Receipt: `THREE_GROUP_SHARED_DAG_V1_RESULT.json`; numerical checks and benchmark: `check_three_group_shared_dag_v1.py`.
