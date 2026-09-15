# Setting2 regional head9.8 QK1 late-group fold V1 preregistration

`ACTIVE_TRACK: WEIGHT_FOLDING`

The exact 289-term QK1 carry-source census is distributed at individual-pair grain: its top ten leave `.66738` replay error and no term exceeds `.04840`. Nine top-ten terms lie within or between attention5 and MLP5/6/7. Freeze the cross-module late group

$$
D=A_5+M_5+M_6+M_7,
$$

and let $R$ be the other 13 carry sources: embedding, attention0–4/6/7, and MLP0–4. Because QK1 is bilinear, the parent carry×carry score has the exact four-block expansion

$$
B(c,c)=B(D,D)+B(D,R)+B(R,D)+B(R,R).
$$

Fold each ordered block through the same fixed QK2 score, native value, head9.8 output projection, residual propagation, MLP16 × MLP17 term, and row-specific UK-minus-US reader.

Reuse all 96 frozen regional rows in exactly 14 length-bucketed forwards. There are 17 reconstructed native carry sources and four grouped ordered terms. No new rows, behavioral logits, fits, backwards, gradients, parameter updates, or quantization.

Predictions:

1. **A — exact instrument.** Carry reconstruction is at most $10^{-6}$, grouped QK1 score and downstream closure are at most $10^{-8}$, and the parent `.8588192173` ratio is reproduced within $10^{-6}$.
2. **B — late self-block is material and stable.** $D\times D$ has change-norm ratio at least `.35` overall and `.20` in each prompt family.
3. **C — three-block compact replay.** The frozen subtotal $D\times D+D\times R+R\times D$ replays the parent paired change with relative error at most `.30`.
4. **D — late self-block leads.** $D\times D$ is the largest of the four blocks globally and in every prompt family.
5. **E — cross-boundary composition matters.** $D\times R+R\times D$ has change norm at least `.25` of the parent.

A pass is a compact attribution grouping on selected rows, not a causal circuit. Its next required test is a fresh selective routing edit comparing the three-block subtotal with $R\times R$, matched QK2, and value-route controls. Failure preserves the four-block census and closes this post-selected late grouping without redefining $D$.
