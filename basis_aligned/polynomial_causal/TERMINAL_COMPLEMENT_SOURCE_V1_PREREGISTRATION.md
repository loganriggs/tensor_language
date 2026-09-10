# Terminal complementary-state sources — preregistration v1

Status: registered preparation; executor extension underway; no native execution or results.
Prior evidence: LEXICAL_FORM_INTERCHANGE_V1_RESULT.json and LEXICAL_FORM_READOUT_FACTORS_V1_RESULT.json. Existing MLP17 dossier, scalar-write and bilinear response precedents checked. This tests a complementary-output source, not the already insufficient last-MLP scalar alone.

Target: explain F-induced lexical drift by separating the last bilinear layer's complementary output from complementary information carried into it. Preserve the existing fixed e direction, A1/A2 rows, four token readers and original-lemma Y form command. No head, direction, rank or threshold tuning. All old capability/selectivity failures remain.

Let h=r+m at the last residual addition and P=I-ee^T. For baseline 0 and F intervention 1, form h_ij=e(e^T h_1)+P r_i+P m_j. Recompute the physical final RMS normalization and score cap for every state. Compare lexical-margin changes against the actual F effect separately for FB/B and FX/X in both frames.

- pred_a: instrument holds. Own-cache/scalar no-op and baseline replay; h_11 full-F bridge; h_10 matches native F with last-MLP output complement restored to baseline. State bridge relative error <=1e-5; readout max error <=1e-3 and relative error <=1e-5. Exact registered forward/sequence counts and finite outputs. Fold the perpendicular structured token readers through Down17 onto actual captured product activations; verify against the actual complementary output with numerical tolerances fixed in the runner before execution.
- pred_b: changed MLP complement is sufficient: h_01 predicts lexical-margin drift with relative L2 error <=0.20 in all four cells.
- pred_c: changed carried complement is sufficient: h_10 predicts lexical-margin drift with relative L2 error <=0.20 in all four cells.

Null: neither source alone is sufficient. These are conditional source-sufficiency tests; even a positive result does not establish an independent upstream producer. Both source hypotheses may hold if their effects are redundant.

Plan and literal price: per frame native B/X/Y, F(B)/F(X), two F arms restoring only MLP17 output complement, and one baseline no-op =8 bodies. Total16 forwards,256 sequences, with the existing all36 scalar-write hooks. Capture last-MLP input/products and terminal r/m for saved-state CPU reuse. Original native weights and contexts remain charged; no extraction claim. No optional extra producer arm is included in this registration.

Implementation: scalar_write_network_executor_v2.py extends the frozen v1 only to return raw17 and per-module writes. Register complement-restoration hook before scalar-write hook so the prescribed F scalar remains unchanged. Final runner source, artifact bindings, product-fold tolerances and dry-run counts must be frozen and checked before managed lane1 enqueue. No GPU execution authorized by an unreviewed draft.

Pre-execution completion: product-reader FP32 bridge uses elementwise absolute tolerance 1e-4+1e-5*abs(reference); all elements must pass. Capture32 input/product visits plus16 complement-hook visits. Sixteen additional physical readouts; saved states include r,m,u,products. Executor, legacy helper, prior states and native source hashes bound before enqueue.
