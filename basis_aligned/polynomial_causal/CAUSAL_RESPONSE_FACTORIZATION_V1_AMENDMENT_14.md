# Causal-response factorization v1 — Amendment 14: training-frontier candidate freeze

Status: prospective before opening any validation artifact or implementing a validation
loader. The complete FIT grid and deterministic FIT analysis already exist.

The validation candidate library is the union of the complete healthy three-seed
rank-pair frontiers under:

1. $(P,C,\text{median FIT MSE})$; and
2. $(P,C,\text{median FIT MSE},\text{median worst-owner-pair NRMSE})$.

No knee, seed, family, or semantic interpretation is selected. Every frozen rank pair
contributes all three optimizer seeds. A rank pair with any failed or unhealthy seed
is ineligible. The current two frontiers happen to be identical; the freezer computes
their union rather than assuming that fact.

The create-only freeze manifest binds the exact published FIT analysis, grid terminal,
source closure, rank-pair identities, and byte/hash identity of every candidate program.
It contains no document response, factor re-fit, validation loader, EVAL capability, or
candidate score. A future validation lifecycle must replay this manifest before opening
the 114-document role and must score every frozen program under the already registered
unconditional and 2/4/8/16-arm calibrated protocols.

This freeze is anti-selection infrastructure, not scientific progress by itself. It
does not establish hierarchy, held-out transport, block-relative validity, semantic
atoms, composition, extraction, removal, OOD behavior, or ledger credit.
