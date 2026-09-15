# Subject-number donor-free head-response proxy V1

The selected subject-number coefficient program uses the interaction between the native L11H3 axis coordinate $z_b$ and the exact head-local response $s_b$ to an opposite-number MLP6/7 donor. This assay asks whether a fixed upstream direction prototype can replace the row-specific donor.

For each leave-one-construction-out fold and answer direction $d$, use only rows in the training construction to average the raw MLP8-input displacement caused by the MLP6/7 `YZ` source:

$$
p_d=\mathbb{E}_{i\in\mathrm{train}(d)}\left[x_{i,YZ}-x_i\right].
$$

For each held-out row and E/A/U/W background $b$, define the donor-free response proxy

$$
\hat{s}_b=u^\top\left(H(x_b+p_d)-H(x_b)\right).
$$

Here $p_d$ is fixed before the held-out construction is evaluated; no held-out opposite-number state enters $\hat{s}_b$. For auditing only, compute the exact donor response

$$
s_b=u^\top\left(H(x_{b,YZ})-H(x_b)\right).
$$

Reuse without refitting the corresponding leave-one-construction-out coefficient vector $\beta$ frozen in the V2 discovery result:

$$
\widehat{\alpha}_b=[1,z_b,\hat{s}_b,z_b\hat{s}_b]\,\beta.
$$

The instrument passes only if exact-$s$ replay reproduces the registered V2 joint-interaction cross-construction metrics within $10^{-10}$. The response proxy passes if its prediction of $s$ has cosine at least `.85` and relative $L_2$ error at most `.50`. The program proxy passes if its coefficient prediction has relative $L_2$ error at most `.45`, improves by at least `.10` over the frozen native baseline error `.6093072967652493`, and degrades by at most `.10` from the exact-response oracle error `.37689362716534275`.

This is an opened-authority, outcome-blind cross-construction test. It reads no answer logits, behavioral effects, downstream causal outcomes, or held-out donor state when constructing $\hat{s}$; performs no fit, backward pass, update, or quantization; and opens no fresh rows. Direction-conditioned prototypes are allowed and must be trained only on the opposite construction. Price: one model forward over 96 role sequences, four 1,152-dimensional fold-specific direction prototypes, 1,536 row-wise offline evaluations of the exact registered L11H3 function, and zero fits.

If the proxy passes, freeze all-row direction prototypes and the already-selected interaction for a fresh-authority causal-substitution test. If it fails, do not enlarge the prototype table by lexical item or background subset; move to a vector-valued recipient-side predictive state.
