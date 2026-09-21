**Recomputing the residual state exposes a full-layer intervention gap**

The frozen full-layer replacement fails all four primary10%direct-source-swap effect requirements. Its isolated leading continuation-mode delta still passes all four. Native final-state replay is exact in the instrument; managed runtime4.189s. No parameters were fitted.

| Primary same-token/cohort comparison | Full path effect error | Isolated mode-delta error |
|---|---:|---:|
| FineWeb continuation |14.77%|5.42%|
| FineWeb spaced word |12.27%|8.01%|
| Code continuation |13.78%|3.10%|
| Code spaced word |10.49%|4.80%|

The intervention replaces the scaled MLP16 polynomial source m with another document's same-current-token source. The recipient carry and attention17 output stay fixed:

$$
h'=h+(m_{\mathrm{donor}}-m),\qquad
x'=\operatorname{RMSNorm}(h'),\qquad
r'=h'+\operatorname{MLP}_{17}(x').
$$

Both original and compressed MLP17 are evaluated this way, then passed through final RMSNorm, unembedding and softcap. This is a direct residual-path intervention, not a swap through every MLP16 descendant: attention17 is deliberately held recipient. Unlike the preceding normalized-port test, the two midpoint inputs and the RMS denominator now change consistently with the same raw source substitution.

The mode comparison is a separate intervention: recompute that mode's scalar change under h',m_donor, then inject only its writer-scaled delta into each program's original background. Passing it does not claim that this single mode explains the full source-path effect. That distinction is necessary: the full-path result fails while the isolated component remains stable.

An actual CPU successor audits per-document failures and normalization shifts. Full continuation error exceeds10% in every represented document:29/29FineWeb and16/16code. Worst errors are19.76% and16.20%. The mode is not uniformly preserved onFineWeb (two documents exceed10% in each cohort), though code mode errors stay below the limit. Recipient RMS ratios span.749–1.520FineWeb and.912–1.167code for the primary donor family. These shifts are measured, not established as the cause of the gap.

This result narrows the compression claim: natural-input effect reconstruction and preservation of a selected component do not imply faithful response of the entire replacement to upstream intervention. The existing exact shared1152product compiler already implements the original two scalar source reads; recreating it would not solve this surrounding-path mismatch. Likewise, the passing mode cannot justify adopting the full replacement as a causal circuit.

The remaining structural task is to improve the joint program, with intervention response as an explicit constraint or evaluation, while allowing learned feature directions and graph sharing beyond native channel pruning. A useful next discriminant is whether the intervention discrepancy is already present in small source displacements (Jacobian response) or emerges with large displacement/normalization change. It should precede assuming that more covariance weighting or another rank sweep will repair the mechanism.

[Managed results](FULL_CHANNEL_DIRECT_SOURCE_SWAP_V1.json) · [Document/normalizer audit](FULL_CHANNEL_DIRECT_SOURCE_SWAP_AUDIT_V1.json) · [Earlier normalized-port comparison](FULL_CHANNEL_CONTINUATION_INTERCHANGE_INTERPRETATION_V1.md).
