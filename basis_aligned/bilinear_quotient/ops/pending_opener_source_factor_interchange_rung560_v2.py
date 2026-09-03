#!/usr/bin/env python3
"""R560 v2 shape-only execution repair; scientific protocol is unchanged."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import pending_opener_source_factor_interchange_rung560 as parent


PREDICATE_CONTRACT = {
    "pred_a_exact_instrument": "native replay and factor product are exact",
    "pred_b_fit_selective_source_factor_exists": "one frozen factor arm passes every FIT target and control",
    "pred_c_selected_source_factor_holds": "the FIT-selected arm passes every SELECT target and control",
}
ORIGINAL_EVALUATE = parent.evaluate


def evaluate_selected_split(model, document, ceiling, source_map, split, arms):
    raw, execution = ORIGINAL_EVALUATE(model, document, ceiling, source_map, split, arms)
    if set(raw) != {split}:
        raise RuntimeError(f"R560 evaluator split envelope changed: {sorted(raw)}")
    return raw[split], execution


def main() -> None:
    parent.evaluate = evaluate_selected_split
    parent.main()


if __name__ == "__main__":
    main()
