An exact input-span obstruction for coefficient matching of the CP parent

22 September 2026, 00:44 UTC. This is a numerical bound computed from the full implicit coefficient tensor of each fitted CP512 parent. It is not a sampled contraction bound, and it is not a bound for the original native quartic tensor.

For a fully input-symmetric quartic coefficient tensor T, form the unfolding with its first input coordinate as rows and all other indices, including output, as columns. Its row Gram K can be computed without constructing T. A CP term's symmetrized four-fold product contributes

$$
T_{v,i,jkl}=\frac14\sum_{a,s}C_{va}f_{as,i}
\operatorname{Sym}_3\left(\bigotimes_{u\ne s}f_{au}\right)_{jkl}.
$$

Therefore

$$
K=\frac1{16}\sum_{a,b,s,t}(C_{:,a}^{\top}C_{:,b})
\left\langle\operatorname{Sym}_3(f_{a,\ne s}),
\operatorname{Sym}_3(f_{b,\ne t})\right\rangle
f_{as}f_{bt}^{\top}.
$$

The inner product averages six products of factor dot products. K is1152 by1152; its eigenvalues sum to the exact squared coefficient Frobenius norm. A student with r quartic CP terms depends on at most4r linear input directions, so this unfolding has rank at most4r. The omitted eigenvalue tail of K is consequently a lower bound on its squared coefficient reconstruction error, regardless of initialization or optimization.

| CP terms | Maximum input rank | Relative coefficient error lower bound, start1001 | Start1002 |
|---|---:|---:|---:|
|64|256|24.219%|22.920%|
|128|512|10.253%|9.372%|
|192|768|4.255%|3.811%|
|256|1024|1.273%|1.135%|
|288 or more|1152|0 from this bound|0 from this bound|

At least265/261 CP terms are necessary for1% coefficient error. These counts are necessary, not sufficient. A zero bound at288 says this particular input-span argument stops excluding a fit; it does not promise that288 products of linear forms can represent the parent.

Redteam: five structural families compare K with the explicitly materialized symmetric tensor's unfolding in d4. Native parent trace replays to1.4e-15relative against the independently implemented CP coefficient Gram. Both computed spectra are positive. This is a floating-point spectral calculation, not an interval-certified theorem about the stored decimal results. The mathematical rank/tail implication itself does not rely on random probes.

Why this does not cancel the queued experiment: its registered parent fidelity bar is1% under the shifted Gaussian, not1% coefficient Frobenius error. Input distributions change which discrepancies matter. The result does prevent interpreting a Gaussian pass as full coefficient equality, or blaming an optimizer for failure at an impossible1% coefficient target with256 terms. The mixed loss will still trade coefficient and Gaussian errors.

These parents already approximate the native tensor poorly in global coefficient space. Their bounds do not transfer as lower bounds for the original model, and no conclusion about selective removal or OOD behavior follows. They also do not rule out other arithmetic DAG architectures at the same operation count.

Decision: retain the registered256-term moving-direction experiment unchanged, report exact parent coefficient error beside Gaussian error, and preserve native response tests. Do not add a1% coefficient gate after seeing this bound. Larger-term experiments, if later warranted, must justify their cost against the shared1088-product program rather than treat288 as a guaranteed solution.

Artifacts: [numerical bounds and dense controls](CP_PARENT_INPUT_CAPACITY_V1.json), [implicit Gram](cp_input_mode_gram.py), [audit](audit_cp_parent_input_capacity.py).
