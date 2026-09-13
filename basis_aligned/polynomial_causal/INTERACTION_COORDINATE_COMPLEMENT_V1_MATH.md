# A coordinate correction instead of a sequence of reflections

13September2026. A second compact adapter represents the same optimal projected interaction with slightly fewer stored scalars and a simple matrix-multiply execution. It passes algebraic and FP32 readout controls. Its nonorthogonal coordinates require a different metric for any subsequent sparsification.

## Construction and dimensions

As before, unfold the mixed tensor into $M\in\mathbb R^{1152\times1536}$, and let $N\in\mathbb R^{1152\times k}$ contain the discarded orthonormal input directions. Its projected approximation is $\widehat M=(I-NN^T)M$.

Choose$k$coordinate indices$A$such that$N_A$is invertible; the remaining$r=1152-k$indices are$B$. Pivoted QR of$N^T$selects these indices, without changing the discarded subspace. Define

$$
C=N_BN_A^{-1}\in\mathbb R^{r\times k},\qquad
J=\begin{bmatrix}-C^T\\I_r\end{bmatrix}\in\mathbb R^{1152\times r}.
$$

Coordinates are ordered as$(A,B)$in this expression. Since$N^TJ=0$, the columns of$J$span the retained subspace. The bottom identity block implies

$$
F=\widehat M_B,\qquad \widehat M=JF.
$$

An input is read through

$$
u=J^Tx=x_B-Cx_A,
$$

then the dense core$F$produces the output/head coefficients. The correction uses one matrix multiplication and a subtraction. Store$rk$correction entries,$1536r$core entries and a144-byte bitmap indicating the coordinate split; the identity and index ordering are implicit. Matrix inversion is not executed at inference; preparation uses a linear solve.

## Actual-weight controls and price

| Coefficient error ceiling | Retained core readers | Local bytes | Storage saving |
|---|---:|---:|---:|
|2%|1105|6,997,004|1.14%|
|5%|1008|6,773,904|4.29%|
|10%|869|6,322,988|10.67%|

All three are cheaper than the corresponding reflector representation. The actual projected-tensor errors are unchanged:1.994%,4.979%and9.999%. Full coefficient and readout replay errors are below1.91e-14. FP32 execution on17fixed independent input/head pairs differs from its FP64 reference by at most1.07e-6relative. This is a readout precision check, not native behavioral preservation. Preparation/control took0.75seconds on two CPU threads.

All native input coordinates remain dependencies. The reduced core has fewer readers, but the correction adds its own computation. No runtime advantage, native regional/FineWeb preservation, global node sparsity or whole-model compression is established. Irregular scalar sparsity still gives greater storage savings at10%coefficient error.

## Executed red team: the reader metric is not Euclidean

Unlike the reflector representation,$J$is not orthogonal:

$$
J^TJ=I_r+CC^T.
$$

If a sparse core introduces error$E=F-\widetilde F$, its actual coefficient error inside the retained subspace is

$$
\|JE\|_F^2=\|E\|_F^2+\|C^TE\|_F^2.
$$

The discarded-subspace error remains orthogonal to$JE$, so it can be added to this **weighted** squared error. Using only$\|E\|_F$would underprice approximation damage.

This was tested by pruning each of the three cores to1%,5%and10%ordinary core error. The weighted formula agrees with explicit reconstruction to3.7e-15relative. Actual extra error is amplified by about1.83×,2.77–2.81×and3.63–3.65×across the three projections. For example, pruning the10%-projection core to10%ordinary core error yields33.1%total target error, not a10%-accurate tensor. These deliberately unadopted probes demonstrate the metric issue; they are not a failed attempt at correctly weighted sparse optimization.

The correction matrices have spectral norms21.3–41.2, so this amplification is compatible with their geometry. The FP32 dense-core control nevertheless passes. Nonorthogonality does not invalidate the dense representation, but future sparse support selection and coefficient refitting must account for$I+CC^T$and preserve the total error budget.

Next useful promotion is native conditional behavioral validation of a frozen dense-core candidate. Weighted sparsity can follow if that representation preserves the circuit's effect; the current evidence alone does not establish the requested behavioral properties.

[Construction and execution](interaction_coordinate_complement_v1.py) · [Price/precision receipt](INTERACTION_COORDINATE_COMPLEMENT_V1_RESULT.json) · [Metric control code](interaction_coordinate_metric_v1.py) · [Metric results](INTERACTION_COORDINATE_METRIC_V1_RESULT.json).
