# Exact-RMS upstream closure preflight fails at the residual composition

The first implementation of the registered residual6-to-attention8 port closure tried to add the full native MLP7 bilinear output to the residual6-derived attention7 state before generating head8. The local preflight is a useful negative result: off-support movement is exactly zero and all outputs are finite, but the maximum attention8 write error is **1.34** relative on the existing 40-fixture panel, far above the `.05` prediction gate.

This is not a fresh behavioral result and does not change the coupled value operator. It says the saved attention7/readers boundary does not expose enough information to add the full MLP7 output in that location using the naive residual formula. The earlier approximate route remains invalid for port closure, and the exact upstream registration stays open.

The failure narrows the next fold: recover the precise block7 residual/mixing convention and compare its generated `other_sources`, MLP7 readout channels, and RMS against an independently captured native mixed8 state before attempting another full-suffix intervention. No source partition or fitted correction will be promoted from this preflight.

## Receipts

- [Exact upstream preflight result](../../CITY_ATTENTION7_EXACT_UPSTREAM_V1_RESULT.json).
- [Exact upstream implementation](../../city_attention7_exact_upstream_v1.py) and [checker](../../check_city_attention7_exact_upstream_v1.py).
- [Port-closure preregistration](../../CITY_COUPLED_UPSTREAM_PORT_V1_PREREGISTRATION.md).
