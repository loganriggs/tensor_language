# Successor pointer interaction low-rank V3 serialization correction

V2 corrected direct-readout geometry and passed every FIT instrument, selecting rank 8. HOLDOUT opened and its native capability, targets, direct readout, and joint-family reports were serialized. The length-seven month joint arm missed its frozen projection bar (`0.466245 < 0.50`), so low-rank transfer was invalidated at the parent-mediator gate. However, V2 serialized the combined HOLDOUT instrument Boolean without the already-computed HOLDOUT self-replay and all-module-ceiling errors, preventing an independent audit from proving that the month projection was the sole failure.

V3 changes only result serialization to include those two HOLDOUT errors. Execution, rows, arms, SVD, ranks, selection, bars, price, and terminal logic remain identical. V1 and V2 low-rank outcomes remain invalid; V3 is used only to audit and register the narrow length-seven parent-mediator boundary if all other instrument components pass.
