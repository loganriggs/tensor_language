"""Outcome-blind parent/cache audit for attention5 leader transfer."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    p = root / "basis_aligned/polynomial_causal"
    bq = root / "basis_aligned/bilinear_quotient"
    result = json.loads((p / "ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_RESULT.json").read_text())
    artifact = torch.load(p / "ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_ARTIFACT.pt", map_location="cpu", weights_only=True)
    caches = {}
    for name in ("final_natural.pt", "ood_code.pt"):
        path = bq / ".rowcache_terminal_copy_induction_v2" / name
        obj = torch.load(path, map_location="cpu", weights_only=False)
        rows = obj["rows"] if isinstance(obj, dict) else obj
        caches[name] = {"shape": list(rows.shape), "sha256": sha(path), "first24_sha256": hashlib.sha256(rows[:24].numpy().tobytes()).hexdigest()}
    leaders = [int(x) for x in result["singleton_order"][:3]]
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pred_a": leaders == [6, 7, 3] and list(artifact["mean_head_input"].shape) == [9, 128] and all(v["shape"] == [192, 257] for v in caches.values()),
        "leaders": leaders,
        "parent_result_sha256": sha(p / "ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_RESULT.json"),
        "parent_artifact_sha256": sha(p / "ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_ARTIFACT.pt"),
        "caches": caches,
        "scope": "Frozen parent leader selection and cache identity only; no new model scores or transfer verdict.",
    }
    out = p / "ATTENTION5_DEVIATION_LEADERS_TRANSFER_V1_CPU_CONTROL.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
