# MLP2 CMR v1: FIT_MEAN findings

## What was computed

MLP2 has 4,608 scalar bilinear products

$$
a_j(x)=(L_jx)(R_jx),
$$

and writes

$$
y(x)=b+\sum_{j=1}^{4608}D_{:j}a_j(x).
$$

On 30,801 eligible token positions from 192 fresh FineWeb documents, this stage
computed the mean $\mu_j$, variance $\operatorname{Var}(a_j)$, and second moment
$\mathbb E[a_j^2]$ of every product.  It then formed three top-512 control supports:

- **LOCAL:** $\operatorname{Var}(a_j)\lVert D_{:j}\rVert_2^2$;
- **RMS:** $\mathbb E[a_j^2]\lVert D_{:j}\rVert_2^2$;
- **MASS:** $\lVert L_j\rVert_2^2\lVert R_j\rVert_2^2\lVert D_{:j}\rVert_2^2$.

LOCAL estimates the immediate write error of replacing one channel by its mean, if
cross-channel cancellation is ignored. RMS includes the channel's constant mean.
MASS uses weights only and does not look at natural-text activation frequencies.

The exact native prefix stopped immediately after the MLP2 product capture. It did
not execute layers 3--17 or construct logits. The 48 four-document prefix calls took
10.54 seconds. No targets, loss, KL, accuracy, validation, or replication data were
opened.

## Numerical result

Retaining 512 products prices the compiled MLP2 at 1,770,624 scalar values, or
11.1175% of native MLP2's fixed-grammar values. This would be an 88.8825% local
MLP2-value reduction if the later finite behavioral gates pass.

| Quantity | Result |
|---|---:|
| LOCAL--RMS top-512 Jaccard | `0.90335` |
| LOCAL--MASS top-512 Jaccard | `0.34737` |
| RMS--MASS top-512 Jaccard | `0.33681` |
| LOCAL--RANDOM top-512 Jaccard | `0.04703` |
| Mean component's fraction of total RMS singleton score | `0.03163` |
| LOCAL score captured by LOCAL512 | `0.21563` |
| RMS score captured by RMS512 | `0.22281` |
| MASS score captured by MASS512 | `0.17728` |

The 512 LOCAL products contain about 21.6% of summed LOCAL singleton score while
occupying 11.1% of products. There is real concentration, but not an extreme sparse
collapse. A random 512-support captures about 10.9% of LOCAL score.

LOCAL and RMS are almost the same selector because product means account for only
3.16% of the summed RMS score on this fit role. The mean fold is still operationally
important: omitted means become the constant bias correction

$$
b' = b + \sum_{j\notin K} D_{:j}\mu_j,
$$

which costs only one ordinary 1,152-value bias already included in the price. But the
small mean-energy fraction explains why adding means barely changes which products
are ranked highly.

MASS is materially different. Weight magnitude alone does not recover the
natural-text LOCAL/RMS ordering. This makes MASS a useful equal-price negative
control, not a substitute for measuring downstream consequences.

All three scores replayed exactly under independent reciprocal rescaling of each
native product channel: top-512 Jaccard was `1.0` and maximum relative numerical
score error was below `4.4e-16`. Thus these scores measure a property of the folded
function rather than an arbitrary choice of native channel scale.

## What this does and does not establish

This result establishes the fit means and control supports needed by the registered
experiment. It does **not** establish that MLP2 is compressible to 512 products.
Summed singleton scores ignore correlations and cancellation between omitted product
writes; RMSNorm, later attention, and later MLPs can amplify or suppress the joint
perturbation. Only the frozen SUFFIX selector followed by actual finite final-logit
CE, KL, NRMSE, top-1, collateral-cell, edit-direction, and margin-certificate tests
can promote the candidate.

The strict whole-model ledgers therefore do not move. The useful new prediction is
narrower: RMS is likely a redundant near-copy of LOCAL, MASS remains a genuinely
different control, and a successful 512-product result must come from downstream
response information or joint cancellation—not from an already obvious ultra-sparse
local energy spectrum.

## Provenance

- Source commit: `27afa8a19f0d2d5268dfbc5d1b85b733b8266276`
- Bundle SHA-256: `043bb52b9580d9c9c342460e5bb80ff579db01486b3b6c6672bf5fba77e46f8e`
- Result SHA-256: `65c1ee33f0399d6489cae0227442d479a9d59b9be98f619d92423cfd39fc7833`
- Receipt-file SHA-256: `9dc14d909a1b4aafd33c67dc7a3d066db4ccc9cb83c7059fe7aaf499ca9e5efa`
- Attempt-1 import failure: `mlp2_cmr_v1_fit_mean_attempt1_failure.json`

