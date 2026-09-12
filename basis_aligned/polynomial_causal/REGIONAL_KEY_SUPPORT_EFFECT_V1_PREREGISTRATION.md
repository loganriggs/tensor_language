# Downstream readout of frozen key-support errors

Unchanged forward13/backward13 supports from original four cue prefixes. The
corrected fresh32 panel changes following queries but duplicates two selection
key prefixes; this is a basic faithfulness/control test, not geographic OOD.
Use the existing five-arm first-value branch executor with native queries:
base, nativefirstswap, forward13key swap, backward13key swap, all-updatekey swap.
Reconstructed cue-key inputs from the saved prefix ports are the only change.
Keep both QK factors joint and every downstream native normalization/write.

A: native producer/attention/branch replay and all-updatekey write replay <=1e-5.
B: A plus forward13 first-value write error <=.1.
C: A/B plus regional signed effect error <=.1, control effect RMS <=.05 native
margin RMS, control mean absolute effect <=.25 native regional effect RMS, no
capable control answer flips. These are the earlier branch fidelity bars.
Backward support remains diagnostic. Preserve earlier key/score failures even
if downstream weighting improves fidelity. No support refit or post-selection.
Native-generated prefix ports, queries and background remain charged.

Price6 native batches32rows through fullbody,5 suffix arms,180sec managed cap,
~1.7MB artifact. No new model fitting. Geographic downstream validation only
follows a passing basic screen. This is a registered implementation in progress;
not evidence that the native experiment has run.
