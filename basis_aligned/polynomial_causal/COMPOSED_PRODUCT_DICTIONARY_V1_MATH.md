# Product-node pruning after composing the private correction

13 September 2026. Target $J_{10}B_9(z)$ from the preceding
[output-spectrum test](COMPOSED_PRIVATE_OUTPUT_SPECTRUM_V1_MATH.md).
This permits many output directions but fewer scalar product nodes. It differs
from the earlier `INTERACTION_PRODUCT_NODES_V1` test of $K(Jz,y)$ with fixed
output vectors: here every retained product has a freely refitted output vector.

Let $C=J_{10}D_9$, and let $H$ be the exact symmetric-product Gram defined in
the spectrum note. For retained product indices $S$, least squares gives

$$
H_{SS}\widehat C_S^T=H_{S,:}C^T.
$$

Rank supports by each native term's squared coefficient norm
$\|C_{:k}\|^2H_{kk}$. A matched control ranks using $D_9$ before composition.
These rankings are heuristics; all output coefficient solves on their supports
are exact up to FP64 roundoff. Normal-equation errors are below
$3.8\times10^{-15}$; retaining all 4608 products recovers the original output
weights within $4.1\times10^{-15}$.

| Retained products | Before-fold ranking error | Composed ranking error | Local scalar saving |
|---|---:|---:|---:|
| 1024 | 83.72% | 82.49% | 79.49% |
| 2048 | 68.91% | 67.40% | 58.97% |
| 3072 | 51.72% | 50.34% | 38.46% |
| 4096 | 28.39% | 27.34% | 17.95% |

The reference stores $J_{10},L_9,R_9,D_9$: 17,252,352 scalars. A standalone
selected-product executor needs $3\cdot1152|S|$ scalars for its input readers
and output vectors. This excludes common normalization, constants and other
branches. If the original MLP9 output is independently required, the joint
program can have a different cost: no whole-model saving is claimed.

## Could better subset search rescue these native readers?

Normalize every scalar quadratic atom to unit coefficient norm. Its Gram is
$\bar H$, and its output vector becomes $A_{:k}=C_{:k}\sqrt{H_{kk}}$.
Let $\mu=\lambda_{\min}(\bar H)$. For any approximation supported on $S$,
the omitted coefficient columns remain $A_{:k}$, so

$$
\|T-\widehat T\|_F^2
\geq\mu\|A-\widehat A\|_F^2
\geq\mu\sum_{k\notin S}\|A_{:k}\|^2.
$$

The best possible omitted sum at a fixed size discards the smallest columns.
The measured full normalized Gram has $\mu=0.802394$. Every 4096-product
subset therefore has at least 24.55% relative coefficient error, regardless
of its refitted output vectors. More strongly, achieving 10% scalar savings
allows at most 4492 products; **every such subset has at least 10.84% error**.
Thus the joint 10%-error/10%-saving criterion cannot be reached merely by
improving support search in this dictionary. These are analytic inequalities
evaluated in FP64, not interval-arithmetic certificates.

The limitation concerns fixed native input products, arbitrary output writers,
this full output scope and this coefficient norm. It does not exclude learned
input readers, block terms, shared nonlinear graphs, or producer/input/output
restrictions. Together with the previous output-rank result, it directs the
next search toward a different representation, not longer fitting of these
same restricted classes. The validated shared-tail executor remains intact.

Code: `composed_product_dictionary_v1.py` (optional `--bound`). Receipts:
`COMPOSED_PRODUCT_DICTIONARY_V1_RESULT.json`,
`COMPOSED_PRODUCT_DICTIONARY_V1_BOUND.json`, and
`COMPOSED_PRODUCT_DICTIONARY_V1_BUDGET_BOUND.json`.
