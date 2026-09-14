"""Post-result numerical classification for the head8.2 source partition."""
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    source = root / "ODD_ATTENTION8H2_SOURCE_EDGE_V1_RESULT.json"
    target = root / "ODD_ATTENTION8H2_SOURCE_EDGE_V1_FP32_CONTROL.json"
    result = json.loads(source.read_text())
    tolerance = 1e-5
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "registered_pred_a": result["pred_a"],
        "registered_partition_tolerance": 1e-10,
        "observed_partition_error": result["max_city_other_partition_error"],
        "native_fp32_tolerance": tolerance,
        "partition_within_native_fp32_tolerance": result["max_city_other_partition_error"] <= tolerance,
        "all_replay_terms_within_native_fp32_tolerance": max(
            result["max_frozen_anchor_error"],
            result["max_head_source_replay_error"],
            result["max_city_other_partition_error"],
        ) <= tolerance,
        "scope": "Post-result numerical classification only. The registered pred_a failure and every scientific verdict remain unchanged.",
    }
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
