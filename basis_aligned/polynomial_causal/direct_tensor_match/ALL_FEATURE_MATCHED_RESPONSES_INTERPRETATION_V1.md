# Do all output features preserve finite changes?

22 September 2026, 02:05 UTC. This audit broadens the earlier root1 newline diagnostic to all 16 output coordinates and many token types. It evaluates frozen CP parents and rank-256 full/lean conditional programs on already-opened states. No new fit, candidate selection or held-out claim is involved.

We group the 16,384 states by current token ID and position, shuffle each group with fixed seed 2209220203, and pair consecutive states. Each pair comes from different documents. A token state appears in at most one pair. This yields **2,494 pairs across 256 documents and 263 token IDs**. Documents still appear in many pairs, so the pairs are not statistically independent. Pair membership uses no reference or candidate values. Matching token and position controls those attributes, but does not isolate a semantic change; the preceding contexts differ.

For pair (a,b), the reference change in the selected native quartic numerator is

$$
\Delta Y=F(x_b)-F(x_a),\qquad
\Delta\widehat Y=\widehat F(x_b)-\widehat F(x_a).
$$

We report the norm of their difference divided by the norm of the reference changes. This tests finite response prediction of the selected polynomial. It does not install a replacement inside the transformer, include final normalization or softcapping, or prove selective manipulation.

| Program | Pooled response error | RMS of relative errors over smaller coordinates 4–15 |
| --- | ---: | ---: |
| CP parent, seed 1001 | 11.60% | 58.09% |
| Full conditional, seed 1001 | 11.72% | 58.66% |
| Lean conditional, seed 1001 | 11.74% | 58.67% |
| CP parent, seed 1002 | 11.98% | 59.60% |
| Full conditional, seed 1002 | 12.08% | 60.23% |
| Lean conditional, seed 1002 | 12.10% | 60.23% |

The three dominant coordinates have about 9–13% response error; coordinate 3 has about 23–25%; coordinates 4–15 have about 42–73%. The smaller programs therefore mostly inherit the original parents' response weaknesses. Their slightly improved pooled *values* do not yield improved pooled *changes* on this matched set.

To distinguish magnitude mistakes from direction mistakes, the JSON also records the share of each coordinate's reference-change squared magnitude assigned the wrong sign. For the lean programs, smaller coordinates have about 2–14% of their change energy on sign errors. This is energy-weighted, not the fraction of pairs with a wrong sign. Weak changes do not dominate this statistic as they could dominate a raw sign count. It still does not identify a semantic category of failure.

The audit checks matching, distinct documents, disjoint state usage, shapes and records candidate and token-artifact hashes. Runtime was 3.46 seconds on two CPU threads. The comparison is descriptive and post hoc; no new adoption threshold or independence-based confidence interval is invented. Existing native removal tests and balanced-feature fitting remain the registered next decisions.

[Script](audit_all_feature_matched_responses.py) · [Results, per-coordinate errors and exact pairs](ALL_FEATURE_MATCHED_RESPONSES_V1.json).
