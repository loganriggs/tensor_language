# Explicit-state executor native replay

Frozen implementation target: `extracted_circuits/crossfirst_state_executor_v1/execute.py` computes the all-source Q7/head8.2-first/MLP8/selected-head9.8 field from three normalized state ports, two norm ports, token IDs and declared weights. It does not receive Q7 or attention-pattern caches. Native prefix and suffix remain external.

Use96existing regional rows (`FIRST_TOKEN_PATH_FRESH_V1_ROWS.json`) and64existing FineWeb controls (`FIRST_TOKEN_REMOVAL_NEWLINE_FRESH_V1_ROWS.json`),160prefixes up to248tokens. Native and all-source-removal arms:320fullforwards,180-second cap. No new outcome selection or data fitting.

A: native regional margin and FineWeb CE/margin anchors replay within1e-4relative. B: assembled field versus independent native-hook computation within1e-4relative over each panel, with individual token-input and Q7 reconstruction errors reported. C: assembled physical removal regional margins and FineWeb CE/margins replay the prior all-source removal within1e-4relative. All must hold before replacing any validated executor.

The independent field uses native MLP7 output-minus-bias projected into the four readers and native head8 normalized/rotated QK arguments with native shared first values; it does not call the assembled field implementation. Head9's already validated selected routing remains a shared fixed primitive. Report this shared dependency rather than claiming two wholly independent implementations.

Null: an input normalization, rotary rounding, mixing, projection, indexing or interface mistake prevents assembled replay. A failed instrument is repaired before any new scientific inference; prior failed sign/control criteria remain failed. This is implementation validation and interface clarification, not OOD confirmation or autonomous circuit extraction.
