#!/usr/bin/env python3
"""Versioned retry of compiler-v2 preflight after config-API source failure."""

from __future__ import annotations

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import early_mlp_state_complete_compiler_v2_preflight as protocol  # noqa: E402


protocol.RESULT = protocol.BQ / "early_mlp_state_complete_compiler_v2_preflight_r2.json"
protocol.MANIFEST = (
    protocol.BQ / "early_mlp_state_complete_compiler_v2_preflight_r2_manifest.json"
)
protocol.LOCK = Path(
    "/workspace/runs/.early_mlp_state_complete_compiler_v2_preflight_r2.lock"
)
protocol.OUTPUTS = (protocol.RESULT, protocol.MANIFEST)
protocol.SOURCE_CLOSURE = (
    Path(__file__),
    HERE / "test_early_mlp_state_complete_compiler_v2_preflight_r2.py",
    *protocol.SOURCE_CLOSURE,
)


def main() -> None:
    protocol.main()


if __name__ == "__main__":
    main()
