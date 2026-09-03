#!/usr/bin/env python3
"""Managed two-circuit GPU smoke for rung 526's batched-gradient path."""

# BQGATE: EXPERIMENT

from mlp0_circuit_response_operator_quotient_rung526_run import main


if __name__ == "__main__":
    main(["--gpu-smoke"])
