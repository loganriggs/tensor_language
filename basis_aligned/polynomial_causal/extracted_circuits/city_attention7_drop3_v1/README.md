# Packed eight-head prefix for city removal

Load attention7.pt, readers.pt, head8.pt with torch.load(weights_only=True), keyed by filename stem. Import execute and call execute.execute(program, residual6, token_ids, city, destination).

Only these files and PyTorch are required. CPU float32 residual6[1,T,1152], integer token_ids[1,T], city index, Boolean destination[T]; returns FP64 attention8 removal delta. Batch1 only; unsupported tokens fail. Frozen vocabulary386 tokens.

Attention7 retains all heads except zero-based head3, with physically packed Q1/K1/Q2/K2/value/output maps. Five MLP7 readers and head8 factors are unchanged. Full first-value table retained for downstream inherited values. RMS8 approximated from non-MLP7 sources; MLP7 remains in all numerators. No fitted proxy or quantization.

Price:22,418,566 FP32 values,89,674,264 bytes;36,864 supplied native floats atT32. Saving884,736 values versus the nine-head generator on the same token vocabulary. Native blocks0–6 and MLP8/suffix remain external. Fresh prediction and selective removal pass. Extraction status/evidence is in manifest. Composition and matched-effect simplicity remain unresolved; this is not a complete circuit or token-only model.
