"""Bounded prior-result audit before a new gerund upstream-producer experiment."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


FILES = (
    "TOKEN_CONTEXT_SOURCE_V1_RESULT.json",
    "TOKEN_CONTEXT_BROADCAST_V1_RESULT.json",
    "LEXICAL_FORM_INTERCHANGE_V1_RESULT.json",
    "TERMINAL_COMPLEMENT_SOURCE_V2_RESULT.json",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    docs = {name: json.loads((root / name).read_text()) for name in FILES}
    singleton = docs[FILES[0]]
    top = {}
    for family in ("A1", "A2"):
        cells = [(name, arm["gate_transfer"], arm["gate_error"])
                 for name, arm in singleton["reports"][family]["arms"].items()]
        top[family] = sorted(cells, key=lambda x: x[1], reverse=True)[:5]
    common_top5 = sorted(set(x[0] for x in top["A1"]) & set(x[0] for x in top["A2"]))
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_shas": {name: sha(root / name) for name in FILES},
        "singleton_top5": top,
        "common_top5": common_top5,
        "max_singleton_transfer": {family: top[family][0][1] for family in top},
        "prior_predictions": {name: docs[name]["predictions"] for name in FILES},
        "closed_repeats": [
            "all36 singleton attention/MLP output-source scan",
            "attention0 first-value-only broadcast",
            "independent lexical/form command axes",
            "MLP17-only versus carried-only terminal complement",
        ],
        "nonduplicative_next_operator": "cumulative depth-interval interchange of the full contextual state feeding MLP17, evaluated on the already causal token/context readers and norm moments",
        "why": "No singleton exceeds .371 transfer, but attn11 and mlp16 recur in both top-five lists and carried-only complement is closer than MLP17-only. A cumulative interval can test distributed buildup without selecting a singleton subset or retrying first-value broadcast.",
        "pred_a": max(top["A1"][0][1], top["A2"][0][1]) < .40 and {"attn_11", "mlp_16"}.issubset(common_top5),
        "scope": "Read-only result audit; recommends an operator class, not a gerund producer or circuit verdict.",
    }
    out = root / "GERUND_UPSTREAM_PRODUCER_PRIOR_ART_AUDIT_20260914_RESULT.json"
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
