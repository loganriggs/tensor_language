from __future__ import annotations

import early_mlp_state_complete_compiler_v2_preflight as failed
import early_mlp_state_complete_compiler_v2_preflight_r2 as retry


def test_retry_uses_isolated_outputs_lock_and_closed_wrapper_source() -> None:
    protocol = retry.protocol
    assert protocol.RESULT.name.endswith("preflight_r2.json")
    assert protocol.MANIFEST.name.endswith("preflight_r2_manifest.json")
    assert protocol.LOCK.name.endswith("preflight_r2.lock")
    assert protocol.RESULT != failed.BQ / "early_mlp_state_complete_compiler_v2_preflight.json"
    names = {path.name for path in protocol.SOURCE_CLOSURE}
    assert "early_mlp_state_complete_compiler_v2_preflight_r2.py" in names
    assert "test_early_mlp_state_complete_compiler_v2_preflight_r2.py" in names
