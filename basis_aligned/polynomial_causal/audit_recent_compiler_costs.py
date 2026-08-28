#!/usr/bin/env python3
"""Reprice S1751/S1752 without loading rows or the model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from compiler_program_cost import CompilerProgramCost


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "bilinear_quotient" / "ops"
RANK_SWEEP = ROOT / "downstream_rank_sweep_results.json"
NONLOCAL = ROOT / "nonlocal_program_class_results.json"
OUTPUT = HERE / "program_cost_audit_2026-08-28.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cost(rank: int, blocks: int) -> CompilerProgramCost:
    return CompilerProgramCost(
        site_count=36, active_token_types=5419, vocabulary_size=50257,
        input_blocks=blocks, rank=rank, input_dim=1152, output_dim=1152,
    )


def efficiency_receipt(item: CompilerProgramCost, recovered: float) -> dict[str, object]:
    factor = item.nats_per_million(recovered, denominator="factor_only")
    conditional = item.nats_per_million(recovered, denominator="conditional_values")
    hook_added = item.nats_per_million(recovered, denominator="hook_added")
    return {
        "recovered_nats": recovered,
        "cost": item.receipt(),
        "nats_per_Mreal": {
            "factor_only_incremental": factor,
            "conditional_active_table_plus_factors": conditional,
            "current_hook_added_tensors_excluding_native_model": hook_added,
            "standalone_zero_native_call": None,
        },
        "factor_only_to_conditional_overstatement_ratio": (
            factor / conditional if conditional != 0 else None
        ),
        "standalone_status": (
            "not available: hook executes native module at every position and uses "
            "native output for uncovered token IDs"
        ),
    }


def main() -> None:
    rank_data = json.loads(RANK_SWEEP.read_text())
    nonlocal_data = json.loads(NONLOCAL.read_text())
    ranks = {}
    for rank_text, row in rank_data["by_rank"].items():
        rank = int(rank_text)
        item = cost(rank, 1)
        # Source serializes Mreals to four decimals, hence at most 50-real rounding.
        if abs(row["cost_M"] * 1e6 - item.factor_reals) > 50:
            raise RuntimeError("S1751 reported factor count changed")
        ranks[rank_text] = efficiency_receipt(item, row["best"]["skip11000"])
    variants = {}
    for name, row in nonlocal_data["variants"].items():
        blocks = int(row["n_feature_blocks"])
        item = cost(8, blocks)
        if abs(row["cost_M"] * 1e6 - item.factor_reals) > 100:
            raise RuntimeError("S1752 reported factor count changed")
        variants[name] = efficiency_receipt(item, row["recovered"]["skip11000"])
    payload = {
        "status": "cost-axis correction; fidelity receipts unchanged",
        "scope": "S1751 and S1752 discovery compilers only",
        "sources": {
            str(RANK_SWEEP.relative_to(HERE.parents[1])): digest(RANK_SWEEP),
            str(NONLOCAL.relative_to(HERE.parents[1])): digest(NONLOCAL),
        },
        "definitions": {
            "factor_only": "trainable low-rank factors; valid only as marginal rank cost",
            "conditional": "active covered-token table values plus factors and shared token IDs",
            "hook_added": "dense allocated tables plus factors, excluding the retained native model",
            "standalone": "requires total input support and zero native calls; current artifacts fail",
        },
        "s1751_rank_sweep": ranks,
        "s1752_nonlocal_variants": variants,
        "conclusion": (
            "factor-only nats/Mreal is not a whole-program simplicity score. The table "
            "dominates conditional storage, and the executed hook retains all native compute."
        ),
    }
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
