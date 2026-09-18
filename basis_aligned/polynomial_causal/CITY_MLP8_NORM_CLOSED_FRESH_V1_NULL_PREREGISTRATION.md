# Fresh same-boundary null for the norm-closed MLP8 mediator

Use the frozen 20-document FineWeb panel from `CITY_MLP8_NORM_CLOSED_FRESH_V1_ROWS.json`. Keep the native attention8 city intervention and full suffix fixed. At the head9.8 value boundary, compare the real norm-closed MLP8 correction against 16 deterministic random corrections. Match each random correction's per-position L2 norm and zero pattern; use seeds `18104000 + 1000*k + context_id`.

The native arm, city-swap arm, real-correction arm, and 16 random arms are evaluated on the same 240 probes. Require exact intervention support, finite outputs, unchanged unrelated heads, and 342 block calls. The real correction passes if its endpoint target RMS is at least twice the random median and exceeds every random correction. All four unrelated-reader controls must be at most 0.5 of the real target. Report the full null distribution and any reversed native rows; do not promote uniform selectivity from a global pass alone.
