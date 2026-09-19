# Exact-RMS upstream closure corrected at the residual composition

The first implementation of the registered residual6-to-attention8 port closure tried to add the full native MLP7 bilinear output to both the residual RMS8 state and the head8 numerator. That double-counted the MLP7 contribution. The corrected implementation adds full MLP7 only to the RMS8 denominator; the folded readers continue to supply its q1/k1/q2/k2/value numerator channels. The maximum write error falls from **1.34** to **2.0e-6** on the existing 40-fixture panel.

The corrected fold also passes the existing FineWeb fixture panel. These are confirmation fixtures from an earlier row selection, so the result is opened relative to selection and is not the required new fresh behavioral panel. It does not change the coupled value operator or yet establish full-suffix prediction/selectivity/composition.

The same corrected operator now passes a new outcome-blind FineWeb panel: **40 sequences from 20 documents, 240 rows**, with maximum relative write error **1.61e-6**, mean error **7.02e-7**, and zero off-support write. The fresh native receipt was corrected to measure the same city-removal intervention as the generator; donor replacement remains a separate selective/composition experiment. This establishes the upstream prediction and support gates at the declared residual6-to-attention8 boundary, while the full suffix and the remaining three behavioral properties remain open.

The next fold is now well-defined: run the corrected generator on a new outcome-blind panel, compare its generated `delta` against an independent native capture, then install it before the coupled value operator and run the four gates. No source partition or fitted correction is promoted from this preflight.

## Receipts

- [Exact upstream preflight result](../../CITY_ATTENTION7_EXACT_UPSTREAM_V1_RESULT.json).
- [Corrected FineWeb preflight result](../../CITY_ATTENTION7_EXACT_UPSTREAM_FINEWEB_V1_RESULT.json).
- [Correction note](../../CITY_COUPLED_UPSTREAM_PORT_V1_CORRECTION.md).
- [Fresh outcome-blind result](../../CITY_COUPLED_UPSTREAM_FRESH_V1_RESULT.json) and [fresh row panel](../../CITY_COUPLED_UPSTREAM_FRESH_V1_ROWS.json).
- [Exact upstream implementation](../../city_attention7_exact_upstream_v1.py) and [checker](../../check_city_attention7_exact_upstream_v1.py).
- [Port-closure preregistration](../../CITY_COUPLED_UPSTREAM_PORT_V1_PREREGISTRATION.md).
