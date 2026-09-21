**Longer evaluation contexts do not explain the transfer failure by themselves**

The registered position-stratified screen completed in 20.45 seconds. All four same-prefix 64-versus-256 replay checks pass (maximum relative discrepancy 1.57e-6, threshold 1e-4). Native component and residual reconstruction checks also pass. These controls exclude a large sequence-length implementation discrepancy at the examined prefixes.

The primary prediction was that component-three natural-effect relative error at positions 128–255 would be at least 1.25 times error at positions 16–63, for both fitting arms and both domains. Every comparison fails that prediction:

| Domain | Paired fit: late / early | Cartesian-context fit: late / early |
|---|---:|---:|
| FineWeb | 0.645 | 0.662 |
| Python standard library | 0.920 | 0.911 |

The statistic is the ratio of RMS relative errors across the two panels, as registered. This rejects the proposed directional explanation; it does not prove that context length never matters, since different positions contain different token/content distributions.

**Successor analysis: absolute error versus the denominator**

The CPU audit obtains exact valid-site counts from the saved donor maps. For each panel and band, it reconstructs error RMS and centered target RMS from the saved error energy and relative error. Target denominators agree across all candidates to 3.51e-16 relative precision.

On FineWeb, late-position absolute error RMS also decreases: late/early ratios are 0.650–0.853 across both arms and panels. Target RMS increases by 1.050–1.208. Therefore the lower relative error is not merely denominator growth.

On code, absolute error RMS increases by 1.252–1.277, but target RMS increases more, by 1.364–1.413. Here the lower relative error conceals some absolute-error growth. This distinction prevents using one normalized statistic as evidence of universal improvement.

The largest relevant early-position warning persists on FineWeb panel two, component three, natural continuation sites: paired error 26.76%, Cartesian error 25.25%, both above 15%. These early sites lie within the historical calibration position range. Matching context length alone is therefore not a justified repair.

**Consequence for the research direction**

Do not launch a longer-context calibration sweep based on the failed hypothesis. The evidence points back to transfer across content/context configurations within the existing position range and the relation between the fitted read errors and their composed effect. A useful next fit must demonstrate document-separated transfer and compare against an equally optimized baseline; fitting a newly observed failure cohort would not provide that evidence. The current seven historical prefix chunks are not seven certified independent documents.

The arbitrary arithmetic-DAG and full folded-target goals remain unfinished. No candidate from this screen is adopted. The panels are already opened diagnostic data, and the new positional breakdown is not independent OOD confirmation.

[Registered plan](POSITION_TRANSFER_PLAN_V1.json) · [Managed result](POSITION_TRANSFER_V1.json) · [CPU audit](POSITION_TRANSFER_AUDIT_V1.json) · [Audit implementation](audit_position_transfer.py).
