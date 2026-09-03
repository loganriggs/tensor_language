#!/usr/bin/env python3
"""Managed two-circuit GPU smoke for rung 526's batched-gradient path."""

# BQGATE: EXPERIMENT
# pred_a: identity-leaf replay, live gradients, and exact contraction pass
# pred_b: same-circuit signatures transfer across held-out documents
# pred_c: frozen pairs transfer to the held-out circuit family
# pred_d: selected donors form repeated groups distinct from rung 525

from mlp0_circuit_response_operator_quotient_rung526_run import main


if __name__ == "__main__":
    main(["--gpu-smoke"])
