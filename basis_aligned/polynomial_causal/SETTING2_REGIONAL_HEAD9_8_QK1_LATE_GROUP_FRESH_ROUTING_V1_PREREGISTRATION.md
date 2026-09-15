# Setting2 regional head9.8 QK1 late-group fresh routing V1 preregistration

`ACTIVE_TRACK: WEIGHT_FOLDING`

The selected-row exact fold defines

$$
D=A_5+M_5+M_6+M_7,
\qquad R=\text{the other 13 layer-0--7 carry sources},
$$

and finds that the three QK1 blocks touching $D$ replay the regional head9.8
carry×carry path with `.11186` error. Test whether that attribution identifies a
causally used routing component on the frozen 48-row `ODD_FRAMING_FRESH_V1_ROWS`
panel, which was not used to choose $D$ or its thresholds. It is fresh relative
to this grouping, though it has appeared in earlier regional experiments.

For every row, recompute the native head9.8 attention pattern and subtract one
specified contribution from the head output before block9 MLP and the complete
native suffix run. Six arms are frozen:

1. native;
2. remove QK1 $D\times D+D\times R+R\times D$;
3. remove QK1 $R\times R$;
4. remove the corresponding three $D$-touching blocks from QK2;
5. remove the current-value contribution projected from $D$;
6. remove complete head9.8.

All edits act at every causal query-key cell. RMS denominators, the unedited QK
factor, first-layer value mixture, other eight heads, block9 MLP, and layers10–17
recompute natively. Score both the row-specific UK-minus-US target reader and the
frozen unrelated control reader. There are 48 rows, two authored template
families, 36 length-bucketed model executions, zero fits, backwards, gradients,
updates, or quantization.

Predictions:

1. **A — exact intervention instrument.** The 17 carry sources reconstruct the
   native carry within $10^{-6}$; a manual full attention9 reconstruction matches
   the native attention write within $10^{-6}$; all values are finite and exactly
   36 executions occur.
2. **B — attribution transfers.** In the native folded reader, the three
   $D$-touching QK1 blocks replay carry×carry with relative error at most `.25`
   overall and `.35` in each family.
3. **C — folded sign predicts the edit.** The negative folded paired change of
   the removed three-block term has cosine at least `.50` and sign agreement at
   least `.70` with its recursively propagated target-logit edit effect.
4. **D — the selected routing edit is material and exceeds the omitted block.**
   Its paired target-effect norm is at least `.10` of complete-head9.8 removal in
   each family, and at least twice the QK1 $R\times R$ effect globally and in each
   family.
5. **E — target-reader selectivity.** For the selected QK1 edit, unrelated-reader
   effect RMS is at most `.50` of target-reader effect RMS in each family.

The QK2 and value arms are mandatory mechanistic controls and are reported without
a null-magnitude prediction, because both may carry separate regional signal. A
pass identifies a fresh, selectively used grouped QK1 routing component within a
native model path. It does not establish that $D$ is an autonomous semantic
module, that the retained path is sufficient for behavior, or that its internal
four-source partition is minimal. Failure closes causal promotion of this
post-selected grouping without redefining $D$ or changing thresholds.
