# Terminal complementary source V2: instrument repair

Post-outcome repair of V1, whose three native FP32 product-reader checks failed. V1 remains invalid and its scientific hypotheses unscored. Same16 forwards256 sequences, fixed rows/e/readers/F commands, all four physical states, .20 source-sufficiency bars and null as V1. No tuning of scientific hypotheses. This rerun is not independent confirmation.

The old product check conflated an exact real-valued weight identity with native FP32 matrix arithmetic and the subsequent FP32 scalar-output edit. New controls separate these:

- pred_a: all old baseline/noop/full/clamped-state and readout controls retain their bars. Capture actual Down output and pre-intervention MLP output; Down output plus bias must equal the pre-intervention MLP output exactly. Compare (C_perp Down) product+C_perp bias against C_perp (Down product+bias), both computed in FP64: elementwise tolerance 1e-9+1e-10*abs(reference). Record native Down and scalar-edit discrepancies separately; these diagnostic quantities do not certify an exact native replacement.
- pred_b: MLP complementary source alone explains F lexical drift with relative L2 error <=.20 in all four cells, conditional on valid instrument.
- pred_c: carried complementary source alone explains F lexical drift with relative L2 error <=.20 in all four cells, conditional on valid instrument.

Literal price:16 native bodies256 sequences;16 input captures,16 product captures,16 native Down output captures,16 pre-intervention MLP captures;16 extra physical readouts and160 FP64 Down control vectors. Entire native model and contexts retained. Native source-sufficiency measurements use actual captured states, not FP64 replacement outputs. Previous capability/selectivity/joint-choice failures remain. Save pre-intervention outputs for CPU diagnosis; freeze source and binding before managed enqueue.
