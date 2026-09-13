# Conditional first-token value path

This package computes the shared first-layer value input change for the head8.2 → MLP8 Q7-interaction → selected head9.8 path. It closes the token-value generator; it is **not an independently extracted circuit**.

`execute.py` computes four first-layer value readings from token IDs and frozen weights, then transports their donor difference through explicitly supplied recipient context. No prompt-specific activation lookup table is used. `program.pt` stores 4,615 scalar coefficients: a 4×1152 value map, two block0 mixing coefficients, four MLP8 eigenvalues, and the head9 block gain. The supplied embedding matrix contains 57,950,208 further weights. Context generation, QK weights, Q7, norms, head9 writer and native suffix remain external and must be charged for an autonomous implementation.

For a paired input differing only at city position c, the measured first-value difference has exactly zero support elsewhere. With j a head9 source position, the scalar write at recipient t is

$$
\Delta w_t=2\alpha_9\sum_{j>c}\frac{\gamma_9(t,j)\gamma_8(j,c)}{\rho_{9,j}s_{8,j}}
Q_j^T\Lambda\big[F(\text{donor city})-F(\text{recipient city})\big].
$$

Here $s_8$ is squared RMS, $\rho_9$ is RMS, and both gamma functions retain the specified joint QK products. The post-city source mask belongs to this tested path definition. The executor handles general aligned token differences, but causal validation currently covers these city pairs only.

[CONTROL.json](CONTROL.json) replays physically tested first-value-only write fields within 4.67e-8 relative error across the four reused constructions. [Physical result](../../ATTENTION8_PHI_VALUE_PORTS_V1_RESULT.json): near-quote −4.88% directed regional transfer,24/24 opposing,31.06% relative effect error versus full head8.2 H-input path. This is a substantial opposing subpath, not a full regional-circuit approximation. No new OOD, broad preservation, or autonomous sufficiency claim.
