# The per-function MDL bill (§1366)

Per-head cost: 884,736 params (6 slices x 1152 x 128). Gates: ~0 bits (token-computable). Probe: 2,304 floats. Route: 18 scalars.

| Kit | Heads | Params | Capability nats | Recovery | Params/nat |
|---|---|---|---|---|---|
| question (16 gated heads + 10.5, clause gate) | 17 | 15.04M | 5.11 | 0.641 | 2.95M |
| comparative (a02 gated + 8.1 + refine {10.5,12.8,11.7,11.6}) | 32 | 28.31M | 5.46 | 0.778 | 5.18M |
| closer (a05 union-gated + 13.8 (brackets + quotes, one kit)) | 55 | 48.66M | 11.23 | 0.644 | 4.33M |
| exclaim (a05 probe-gated + pair (deploy-legal, recall-capped)) | 56 | 49.55M | 4.19 | 0.676 | 11.82M |

Naive sum: 141.56M -> shared-heads-once: 54.86M (saving 61.2%). Unique heads: 62/162.
# The bill, re-priced on the commons (§1369)

- Commons: 22 heads = 19.5M (serves every kit, twice-vetted)
- Specialists: 6 unique heads = 5.3M (10.5, 11.6, 11.7, 12.8, 13.8, 8.1)
- Gates: token-computed, ~0 bits. Route: 18 scalars.
- **Total: 24.8M params (4.5% of the model) for 24.2 capability-nats across four families (66-87% recovery each).**
- 1.03M params/nat — 2.4x better than the §1366 bill.
- Per-capability marginals (own heads + half-share of moonlighters): question 0.44M, comparative 3.98M, closer 0.88M
