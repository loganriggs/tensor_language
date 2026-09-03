#!/usr/bin/env python3
"""Managed exactness/liveness smoke for rung 527's context-source terms."""

# BQGATE: EXPERIMENT
# pred_a: exact centered polynomial, small remainder, native replay, live edits
# pred_b: discovery route remains executable without opening scientific outcomes
# pred_c: the 30 held-out circuits remain unopened
# pred_d: physical substitutions remain unopened
# pred_e: no grouping claim is made by the smoke

from mlp0_centered_context_source_quotient_rung527_run import main


REGISTERED_PREDICATES = {
    "pred_a": "exact centered terms, small remainder, native replay, and live edits",
    "pred_b": "discovery route executes without scoring the scientific pair relation",
    "pred_c": "held-out circuits remain unopened",
    "pred_d": "physical substitutions remain unopened",
    "pred_e": "smoke makes no context-term grouping claim",
}


if __name__ == "__main__":
    main(["--gpu-smoke"])
