# Remaining code failures originate before the final readout

The frozen graph and covariance pair baseline were replayed on the32openedcode files from both panels, using exactly the previous same-token cross-document donor maps. Source reconstruction passes; no coefficients or rows were changed.

For component3/spaced-word, graph-to-baseline scalar-error ratios are1.14124 natural and1.11399 hybrid. The corresponding logit-effect ratios are1.13714 and1.10600. Both already fail1.10 before the final RMSNorm/softcap. Therefore endpoint weighting is not responsible for introducing these threshold failures. This does not assert the endpoint has no effect at all.

The change-error ratio passes at.94265. Exact paired-error identities explain how this can coexist with worse levels. If e_n,e_h are natural and hybrid scalar errors,

$$\langle e_n,e_h\rangle=\tfrac12(\|e_n\|^2+\|e_h\|^2-\|e_h-e_n\|^2),$$

$$\|\tfrac12(e_n+e_h)\|^2=\tfrac12(\|e_n\|^2+\|e_h\|^2)-\tfrac14\|e_h-e_n\|^2.$$

For spaced-word sites, the graph's common-level error is1.23434times baseline, while its change error is.94265times baseline. Natural/hybrid error cosine is.43845 for the graph versus.19740 for the baseline. Some error therefore persists across the donor change and cancels in the difference. This does not imply a constant offset, nor an invariant semantic feature. All-token and continuation-cohort common-level ratios are below1.

Do not repair the final logit readout based on these failures; investigate source-level joint-read error and its correlations. Earlier mean/affine corrections already matched individual quadratic moments, but products of corrected reads can still carry joint errors. Earlier Gaussian moment controls failed, so a raw covariance/Gaussian shortcut is not justified without a new direct check. The next diagnostic should separate the two read-error terms and their product, preserving signed cross terms, rather than infer a constant bias from aggregate energy.

[Managed scalar diagnostic](FRONTIER_SCALAR_DIAGNOSTIC_V1.json), [paired-error identity audit](FRONTIER_ERROR_TRANSPORT_V1.json), [audit code](audit_frontier_error_transport.py). Results are explanatory analyses of already-opened validation data, not new independent validation or a promoted circuit.
