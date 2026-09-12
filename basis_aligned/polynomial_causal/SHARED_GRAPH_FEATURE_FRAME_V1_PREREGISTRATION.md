# Closed feature execution of an existing weight-derived shared subprogram

Freeze SHARED_READER_JOINT_FIT_V1_SPECTRAL_GRAPH.pt. Its original fit remains
unconverged and is not promoted here. Use the union of shared-parent-dependent
terms only: mixed terms plus shared pairs, each once. Private-only quadratics
and original MLP bias are outside this extracted subprogram. Never sum all
standalone node-removal banks; they double-count shared-pair terms.

Undo the output whitener to obtain physical writers. Build a common orthonormal
frame from normalized nonzero left/right reader and physical writer columns,
SVD relative cutoff1e-12. The projected factor bank must replay all factors with
relative error <=1e-10. Bias zero, residual reentry(1,0), native1152D RMS scale.

Choose the off-diagonal shared-parent pair with largest physical pair-writer
norm, using weights only. Test baseline, removal of either parent, and both.
A term is removed if either of its parent dependencies is disabled. Compare
with the original graph evaluated with those parents zeroed minus the same
private-only graph. Positive CP/graph identity <=1e-10 on64seeded vectors.

On72saved native pre-MLP17 inputs, execute the residual subprogram x+G(RMS(x)),
then full50304 unembedding, finalRMS and cap. This is NOT the native fullMLP:
other private/background updates are deliberately outside this mathematical
subprogram, so no native behavioral fidelity is measured by its output logits.

A: frame/factor/graph replay <=1e-10. B: reduced versus direct subprogram full-
logit/features/norm errors <=1e-10 eacharm. C: individual/joint logit-effect
relative errors <=1e-8 and joint-interaction replay <=1e-6, with nonzero effects
and interaction norm >1e-8. Existing dense output weights and initial readout
cache are priced. Compare compact DAG and expanded CP, not just expanded CP.
CPU only, no optimizer/new bodyforward. Compact savedframe/bank artifact only.
