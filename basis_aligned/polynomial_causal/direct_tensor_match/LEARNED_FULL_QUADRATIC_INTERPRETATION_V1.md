**Learned directions improve fitting but degrade the native comparisons**

The100-step full-target pilot compares inherited and random input factors, solving output weights exactly at every step. Both use3686products and14,067,072coefficients after exact affine mean/tangent repair. The inherited start wins by fitting loss. FP32 export drift1.57e-6passes; no native outcomes enter selection.

| Start | Coefficient error | Historical response error |
|---|---:|---:|
|Inherited, before input learning|11.17%|3.53%|
|Inherited, after100steps|10.57%|1.88%|
|Random, after100steps|44.82%|3.54%|

The declared10%relative coefficient improvement fails; the response guard passes. Runtime183.456s for both starts. A100-step pilot is not evidence of global convergence or impossibility for random initialization.

The frozen selected program then fails native full-path and leading-mode gates. Natural aggregate fidelity passes the loose10%threshold but worsens versus the original same-cost affine pruning program on the same documents.

| Primary cohort | Fixed products + response refit: full-path error | Learned products: full-path error | Learned isolated-mode error |
|---|---:|---:|---:|
|FineWeb continuation|14.59%|20.07%|13.98%|
|FineWeb spaced word|11.62%|15.94%|22.80%|
|Code continuation|12.51%|16.36%|6.92%|
|Code spaced word|9.03%|11.23%|6.51%|

Natural fullMLP effect errors are6.64%FineWeb and2.78%code, compared with4.44%/2.12%for the original program. Average CE added is+.00651/+.00406nats/token, compared with+.00426/+.00144. These comparisons reuse the opened panel; they are diagnostics, not independent confirmation. The new program is not adopted, and the original program's fresh validation does not transfer to it.

The implementation checks narrow the failure interpretation: exact dense coefficient gradients, conditional least-squares checks, native endpoint replay and FP32 export replay pass. They do not prove absence of every implementation issue. The observed result is consistent with objective/distribution mismatch and extra freedom to overfit historical geometry, not proof that learned directions cannot work.

An actual CPU successor makes one identification limit explicit. The empirical directional design has2048rows and3686product columns. Even at fixed readers, at least1638coefficient combinations per output are invisible to that response loss—at least1,886,976output-weight directions overall. The exact coefficient term constrains those directions, but low response error alone cannot identify the polynomial. The random-start result illustrates that distinction numerically. This count is a rank upper-bound argument, not a measured exact rank or proof of why the selected program generalizes poorly.

The restricted two-stage toy results remain useful: wide discovery followed by product deletion/refitting reaches planted size in four of five families. They do not rescue the native result. A stronger native search needs evaluation of candidate computations beyond one covariance/finite-response geometry, and explicit preservation of the grounded computation; simply continuing this fitting objective because its loss falls is not justified.

[Fit](LEARNED_FULL_QUADRATIC_V1.json) · [Native tests](LEARNED_FULL_QUADRATIC_NATIVE_V1.json) · [Response-identification bound](FULL_QUADRATIC_RESPONSE_IDENTIFICATION_V1.json) · [Toy two-stage study](LEARNED_FULL_QUADRATIC_TOY_INTERPRETATION_V1.md).
