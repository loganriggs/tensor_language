**The two coefficient-fitted CP dictionaries agree much more on the activation-shaped Gaussian than as global polynomials, and individual terms are not stable.**

This CPU audit runs alongside live mixed-objective feature fitting. It compares the two archived CP400starts1001/1002. All numbers below compare **candidates with each other**, not either candidate with the native teacher. Outputs use the declared16-readout metric. The shiftedGaussian uses the6144calibration-state mean/covariance; the centered functional comparison subtracts the exact mean of each quartic feature, rather than centering the input law.

|Metric|Whole-function cosine|First-to-second relative difference|Second function's best projection error in first dictionary|Matched terms above0.9 /512|Shared subspace directions above0.9 /512|
|---|---:|---:|---:|---:|---:|
|Symmetric coefficient tensor|0.8143|60.96%|55.88%|4|7|
|ShiftedGaussian|0.9930|11.91%|2.55%|15|38|
|Centered shiftedGaussian|0.9916|12.93%|3.52%|13|38|

Projection refits every output coefficient freely in the first dictionary. It is an exact least-squares dictionary projection with numerical rank cutoff1e-10 after normalizing atom norms, not an exported compressed program or held-out prediction. All dictionaries have numerical rank512. Principal angles compare entire scalar polynomial spans; only38directions have canonical correlation above0.9under either Gaussian comparison. High total-function agreement does not mean all directions of those512-dimensional spaces agree.

One-to-one term alignment maximizes **signed output-contribution cosine**: a term includes its16-vector writer and quartic scalar polynomial. Opposite output effects do not count as identical. This handles term permutation, input-slot permutation and compensating factor/readout scaling. Independent small controls replay both those CP gauges and an arbitrary invertible polynomial-dictionary basis change. The latter need not preserve CP atom form; it verifies that the span comparison recognizes an equivalent dictionary even when individual coordinates change.

Interpretation: weak one-to-one alignment alone could have been a basis artifact. The projection test makes the distinction explicit. In Gaussian geometry, a candidate function can be represented well in the other dictionary even though the full dictionaries and individual terms differ. In coefficient geometry, that projection remains poor. Thus current evidence supports approximate function-level agreement under a chosen input law, not stable global polynomial units or semantic circuits. Dense linear recombinations also retain the original product cost unless graph simplification actually removes computations.

No model was changed, adopted or pruned by this audit. Learned mixed-objective results remain pending; rerun the same comparison only after both artifacts are complete.

[Baseline receipt](CP_RESTART_IDENTITY_BASELINE_V1.json) · [CPU implementation](audit_cp_restart_identity.py).
