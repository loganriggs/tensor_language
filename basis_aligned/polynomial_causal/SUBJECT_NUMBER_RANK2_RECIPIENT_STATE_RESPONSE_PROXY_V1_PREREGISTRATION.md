# Subject-number rank-2 recipient-state response proxy V1

The two-vector donor-free MLP6/7 prototype predicts the exact L11H3 response scalar at relative $L_2$ error `.28804`, but its residual is amplified by the required $z\hat s$ interaction and the complete coefficient program fails at `.51585`. Earlier causal evidence says MLP6 and MLP7 must remain one interacting operational source, so this assay adds recipient-side state without splitting the group or introducing lexical/background tables.

For each leave-one-construction-out fold and answer direction $d$, collect the training construction's 8 rows under all 16 E/A/U/W backgrounds. Let $x_{i,b}$ be the recipient/background raw MLP8 input and

$$
\delta_{i,b}=x_{i,b,YZ}-x_{i,b}
$$

be the grouped MLP6/7 displacement. Center the 128 training pairs. Freeze the top two right singular vectors $V_x\in\mathbb R^{1152\times2}$ of recipient state and $V_\delta\in\mathbb R^{1152\times2}$ of displacement. Define the two-dimensional predictive state

$$
r_{i,b}=(x_{i,b}-\bar x)V_x.
$$

Fit only the $2\times2$ ridge map

$$
A=(R^\top R+\lambda I)^{-1}R^\top D,
\qquad
\lambda=10^{-3}\frac{\operatorname{tr}(R^\top R)}{2},
$$

where $R$ contains recipient scores and $D=(\delta-\bar\delta)V_\delta$ contains displacement scores. Predict a held-out displacement by

$$
\widehat\delta_b=\bar\delta+r_bA V_\delta^\top,
$$

then compute

$$
\hat s_b=u^\top\left(H(x_b+\widehat\delta_b)-H(x_b)\right),
\qquad
\widehat\alpha_b=[1,z_b,\hat s_b,z_b\hat s_b]\,\beta.
$$

$\beta$ is the already frozen leave-one-construction-out interaction coefficient vector. It is not refit to the proxy.

The instrument passes if exact-$s$ replay matches the registered interaction metrics within $10^{-8}$ and the mean-prototype proxy matches its registered V2 metrics within $10^{-8}$. The rank-2 response passes if it improves relative $L_2$ error by at least `.05` over the mean-prototype response error `.2880428105400366` and has cosine at least `.95`. The complete program passes if coefficient relative $L_2$ is at most `.45`, improvement over the native baseline `.6093072967652493` is at least `.10`, and degradation from the exact-response oracle `.37689362716534275` is at most `.10`.

This is outcome-blind discovery on the already-open 32-row authority. The four fold/direction maps use only the other construction's rows. The assay performs no behavioral/logit access, downstream causal-outcome access, coefficient fit, backward pass, parameter update, or quantization. It runs one physical forward over 96 role sequences, four rank-2 predictive-map fits, and 2,048 row-wise offline evaluations of the exact L11H3 function. Each map is restricted to two recipient coordinates, two displacement modes, two means, and a $2\times2$ map; no row identifier, lexical identity, or background subset is a feature.

If this fails, reject rank-2 linear recipient-state prediction at this interface rather than increasing rank on the opened authority. If it passes, freeze an all-row map and require fresh-authority causal substitution.
