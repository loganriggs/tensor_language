**Frozen full-layer baseline passes a new document panel**

Both raw and affine-corrected3686product programs pass the registered aggregate fullMLP effecterror<10%criterion on32newFineWebdocuments and16newstdlibfiles. The correction improves eachdomain, and its document-bootstrap95%uppermeanCEdamage remains below.02nats/token. No parameters or selection changed after viewing the panel.

| Program | FineWeb effecterror | Code effecterror | FineWeb CEadded | Code CEadded |
|---|---:|---:|---:|---:|
| Raw,20.009%factor saving |5.294%|3.567%|+.005865|+.004750|
| Affine,11.668%factor saving |4.440%|2.119%|+.004256|+.001436|

Affine effecterror95%document-bootstrap intervals are[3.480%,5.777%]FineWeb and[2.016%,2.227%]code. CEadded intervals are[.002130,.006311]and[-.001186,.004091], respectively; lower is better. These are fullMLP logit-effect reconstruction errors, not language-model error rates. The correction preserves teacher value and gradient at the calibration mean but still incurs positive average CE damage.

The exported FP32 residual-write program stores14,067,072scalars including its denseaffinebranch. Its calibration drift against FP64 is4.72e-7relative. It requires only the normalized1152-dimensional lastMLP input to execute; upstream model computation and the original Downbias remain external. Raw evaluation omits the affinebranch. Common frames and native background are not claimed as savings; no wholemodelruntime measurement was made.

Panel preparation excludes all eight prior document-indexed panels, known cached FineWeb32-token excerpts anywhere in accepted documents, and corresponding codefiles. This is document/file-level separation for this study, not a guarantee against model pretraining overlap, all historic project data, or shared package-level code patterns.

A CPU successor checks per-document regressions and maxima. Its results are retained separately so the aggregate pass is not mistaken for uniform fidelity. No semantic feature identity, selective removal, arbitrary component reuse, or standalone token-to-output circuit has been shown. Retain this as a full-layer compression baseline for the decomposition research, not an adopted interpretable circuit.

[Fresh results and document rows](FULL_CHANNEL_FRESH_V1.json) · [Per-document audit](FULL_CHANNEL_FRESH_DOCUMENT_AUDIT_V1.json) · [Export receipt](FULL_CHANNEL_EXPORT_V1.json) · [Panel provenance](FULL_CHANNEL_FRESH_ROWS_V1.json) · [Program](FULL_CHANNEL_COVARIANCE_PROGRAM_V1.pt).
