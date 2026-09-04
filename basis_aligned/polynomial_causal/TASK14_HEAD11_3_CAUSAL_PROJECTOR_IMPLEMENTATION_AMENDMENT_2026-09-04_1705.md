# Task 14 head-11.3 causal projector — endpoint-replay implementation amendment

Frozen at 2026-09-04 17:05 UTC, before any projector fit or inner-SELECT score. This prospective amendment corrects one implementation and price omission found during independent backend review. It does not change the data split, objective, rank ladder, success bars, or stopping rule.

## Measured endpoint replays

The preregistration requires rank 0 to reproduce native logits exactly and rank 128 to reproduce complete-head-interchange logits exactly. The first backend draft incorrectly filled both health fields with the constant `True` without executing either check.

Before the first fit can be considered healthy, the production backend must now perform both checks on the complete inner-SELECT set:

1. Run the rank-0 projected-interchange path once for each of the 64 unique SELECT endpoints (the union of recipients and donors) and require `torch.equal` against the cached native post-softcap logit vector.
2. Run the rank-128 projected-interchange path for all 145 SELECT relations and require `torch.equal` against the cached complete-head-interchange post-softcap logit vector for that exact relation.

The projected-interchange implementation returns the recipient head vector directly at rank 0 and the donor head vector directly at rank 128. It does not evaluate a nominal empty or identity matrix product whose floating-point roundoff could make an exact endpoint fail. The model is nevertheless rerun through each intervention path, so the checks test the complete hook and downstream execution rather than only the local algebra.

The two measured booleans are cached after the first execution and copied into every fit-health record. A false value makes the fit unhealthy and therefore makes Program A instrument-invalid; it cannot become a scientific null.

## Corrected prospective price

With batch size 32, the rank-0 check costs two forwards and 64 example evaluations. The rank-128 check costs five forwards and 145 example evaluations. They require no backwards pass or stored scientific evidence. The corrected primary ceiling is therefore:

- 1,206 forwards;
- 902 backwards;
- 37,700 example evaluations;
- 141,824 raw tensor bytes.

The first ordinary fit may report at most 120 forwards, 100 backwards, and 3,800 example evaluations because it performs the shared seven-forward replay. Later fits reuse the measured replay receipt. The rank-8 and conditional incremental prices are unchanged because the endpoint checks run only once per Program-A backend.
