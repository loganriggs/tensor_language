# Conditional compression: parent-error compensation and uneven feature fidelity

22 September 2026, 02:02 UTC. Frozen rank-256 full and lean conditional programs were evaluated on the already-opened 16,384 states from 256 documents. This is a diagnostic of the selected 16-output MLP16→17 quartic path, not new held-out confirmation or full-model error. No fitting or selection took place.

Let Y be native values, P the CP parent's values and C the conditional program's values. Define parent residual E=P-Y and compression edit D=C-P. Then

$$
\|C-Y\|_F^2=\|E\|_F^2+\|D\|_F^2+2\langle E,D\rangle.
$$

The cross term matters. An approximate student can match native values better than its parent when its deviations cancel some parent errors. This is a real improvement on the inspected panel, but does not establish an exact native identity or preservation of other responses.

| Program | Native pooled error | Edit norm / native norm | Cosine of parent residual and edit | Documents improved / 256 | Output coordinates improved / 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full, seed 1001 | 7.314% | 1.468% | −0.147 | 136 | 5 |
| Lean, seed 1001 | 7.244% | 1.510% | −0.194 | 164 | 5 |
| Full, seed 1002 | 7.528% | 1.314% | −0.126 | 121 | 3 |
| Lean, seed 1002 | 7.457% | 1.359% | −0.180 | 155 | 2 |

The corresponding parent errors are 7.384% and 7.581%. For both lean programs, the negative cross term has greater magnitude than the added edit energy. The lower pooled score therefore comes from compensation, not uniformly closer matching in every feature. Seed 1001 improves coordinates 0, 4, 5, 10 and 11; seed 1002 improves only 0 and 4. Coordinate 0 is the dominant output. These are fixed basis coordinates, not semantic categories.

Features 4–15 retain roughly 43–70% relative error. The worst 10% of token states carry about 46% of squared error. The worst document has about 52% / 56% pooled relative error for the two lean programs; document-relative percentages can be large for low-energy reference documents. The audit stores document quantiles and per-feature edit/cross terms as well as aggregate error.

This preserves the main claim of the conditional program: a smaller executable approximation with similar aggregate values. It does not upgrade component fidelity, input sensitivity, causal selectivity or composition. Native removal tests remain the relevant next check and are already queued; balanced producer training addresses the separate small-feature capacity question.

Validation: the residual identity replays below 1.1e-18 after division by native energy, all predictions are finite, input/output shapes are asserted, and candidate/parent SHA256 hashes are recorded. Frozen exports are evaluated in float64, without changing the programs used by queued jobs. Runtime was 3.54 seconds on two CPU threads. No causal or statistical independence conclusion follows from these bookkeeping checks.

[Audit script](audit_conditional_residual_accounting.py) · [Full results](CONDITIONAL_RESIDUAL_ACCOUNTING_V1.json).
