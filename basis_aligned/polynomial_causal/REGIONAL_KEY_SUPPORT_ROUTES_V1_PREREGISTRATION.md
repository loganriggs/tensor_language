# Actual-query audit of frozen thirteen-update key supports

The query-independent joint-key norm may over-penalize directions that queries
do not read. Evaluate the unchanged forward/backward13 supports using native
queries on all112 corrected prompts. Neither support is reselected or adjusted.
At each of layers8/9/13 replace only the cue key input with its saved-prefix
background plus selected update differences; all-port reconstruction is the
positive control. Preserve both QK factors, all RMS operations and native RoPE.

Report the relative Frobenius error of the cue routing scores, over all heads
and causal query positions, separately for original, geographic and fresh
panels and separately for each layer. Also report last-query-position error.
The original four prefixes selected the support; geographic eight prefixes
did not. Fresh prompts provide different queries but duplicate original keys.

- pred_a: all-port score replay <=2e-5 in every panel and layer.
- pred_b: pred_a and forward13 original-panel score error <=.1 in all3layers.
- pred_c: pred_a/pred_b and forward13 geographic score error <=.1 in all3layers.

Backward support and fresh-query scores remain diagnostics. A failure only
rejects these supports under actual queries, not all cross-module feature
combinations. This audit does not include downstream value/write weighting or
behavioral effects. Aggregated score error is not a per-head guarantee.
Price18 batches112rows through14blocks, three frozen key variants,180second cap,
no fit, no logits or large new artifact. Bind prior support and prefix receipts.
