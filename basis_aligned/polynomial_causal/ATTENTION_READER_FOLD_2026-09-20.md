# Contextual attention reader and finite query reuse

The exact five-factor attention adjoint replays native suffix gradients. Native attention forward maxabsolute error3.18e-12, backward1.76e-19; saved pre11 reader replay1.63e-19. RMS derivatives, inverse RoPE, both QK factors, value mixing and the fixed first-layer cache are explicit. CPU autograd and directional finite differences independently check the derivative pairing. Implementation: attention_reader_fold.py.

At pre11 the reader decomposes into residual,Q1,K1,Q2,K2,V. Whole-branch omission's worst opened source-gradient relative errors are1.0665,.1484,.3877,.06957,.3101,1.2331 respectively. Only Q2 passes the preregistered10% all-output screen. This is a tangent screen on the full23-direction interface, not an independent circuit.

Native finite test freezes the attention11 query projection at its unedited value during existing23-port oracle-selected edits. Q2 and Q1 are tested separately. All subsequent native computation recomputes. This is a conditional replacement requiring baseline query context, not deletion of the QK product, its weights, source generator or selector.

| Arm | Fidelity cells | Selectivity cells | Worst number error | Worst control error |
|---|---:|---:|---:|---:|
|Full|16/16|16/16|0|0|
|Freeze Q2|16/16|16/16|.010327|.008622|
|Freeze Q1|16/16|16/16|.019789|.020076|

Errors normalized by full number-effect L2. Q2 retention min.93499, maxcollateral.07661; Q1 min.93488, maxcollateral.07892. Previous native effects and baseline-hook replay exactly;12prefix32native suffix calls,1.41s reported run. Inputs were already opened and Q2 selected using their derivatives. No fresh OOD promotion. Q1 was a live comparison, not guaranteed failure; its success is preserved rather than redescribed as a negative control. Full-gradient omission and selected finite-direction omission measure different objects, so Q1's derivative failure is not a contradiction or an implementation bug.

Post-result CPU consequence: audit_attention_reader_context_v1.py exactly partitions canonical reader changes among the six branches, with source-number-matched opposite/congruent donors and recipient source vectors. ATTENTION_READER_CONTEXT_V1.json gives all nine outputs. Number signed shares (residual,Q1,K1,Q2,K2,V): subject unlike_nearby[-.010,0,-.016,0,-.036,1.061]; subject beside_subject[.814,-.015,-.026,-.004,-.004,.235]; attractor unlike_nearby[.388,-.022,.242,-.018,.014,.397]; attractor beside_subject[.477,-.021,.083,-.013,.009,.466]. These sum to1 per group before rounding. Negative or >1 shares represent cancellation. This is algebraic context partition, not separate causal interventions or unique mechanisms. Residual carries downstream context, so the result does not identify attention11 alone as the context generator.

Next discriminator: compose Q1/Q2 freezes; their separate success does not guarantee joint success because the two QK products interact. Then freeze a query-context rule on genuinely new inputs before claiming reuse. Current native model still needed for source edits, context, and per-input selector; no storage reduction or independent effect predictor established. The context partition also redirects later folding toward the value-reader and residual branches rather than arbitrary static mean readers.

Primary receipts: attention_reader_fold_v1_result.json and tensors; attention_q2_freeze_v1_result.json; corresponding managed logs. Preregistrations: ATTENTION_READER_FOLD_V1_PREREGISTRATION.md and ATTENTION_Q2_FREEZE_V1_PREREGISTRATION.md. Entire goal remains active.
