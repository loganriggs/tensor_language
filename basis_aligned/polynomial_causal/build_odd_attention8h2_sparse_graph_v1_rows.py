"""Freeze an outcome-blind natural FineWeb panel for prospective graph selection."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "bilinear_quotient/.rowcache_compiler_v21/fineweb_n192_skip39000.pt"
OUT_ROWS = HERE / "ODD_ATTENTION8H2_SPARSE_GRAPH_V1_ROWS.json"
OUT_CONTROL = HERE / "ODD_ATTENTION8H2_SPARSE_GRAPH_V1_CPU_CONTROL.json"

CITY_PAIRS = [
    (" London", " Boston"), (" Bristol", " Austin"),
    (" Oxford", " Chicago"), (" Manchester", " Seattle"),
    (" Liverpool", " Denver"), (" Glasgow", " Dallas"),
    (" Edinburgh", " Miami"), (" Birmingham", " Atlanta"),
    (" Cambridge", " Houston"), (" Leeds", " Phoenix"),
]
ENDPOINTS = [
    (" colour", " color"), (" centre", " center"),
    (" travelled", " traveled"), (" favourite", " favorite"),
    (" honour", " honor"), (" behaviour", " behavior"),
]
LEFT, RIGHT, CONTEXTS = 12, 20, 8


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if OUT_ROWS.exists() or OUT_CONTROL.exists():
        raise FileExistsError("sparse-graph row outputs already exist")
    enc = tiktoken.get_encoding("gpt2")
    token_pair: dict[int, int] = {}
    region: dict[int, str] = {}
    label: dict[int, str] = {}
    for uk, us in CITY_PAIRS:
        uk_ids, us_ids = enc.encode(uk), enc.encode(us)
        assert len(uk_ids) == len(us_ids) == 1
        ui, si = uk_ids[0], us_ids[0]
        token_pair[ui], token_pair[si] = si, ui
        region[ui], region[si] = "British", "American"
        label[ui], label[si] = uk.strip(), us.strip()
    endpoints = []
    for uk, us in ENDPOINTS:
        uk_ids, us_ids = enc.encode(uk), enc.encode(us)
        assert len(uk_ids) == len(us_ids) == 1
        endpoints.append((uk_ids[0], us_ids[0]))

    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    candidates = []
    for source_row, tensor_row in enumerate(source):
        ids = tensor_row.tolist()
        for position, token in enumerate(ids):
            if token not in token_pair or position < LEFT or position + RIGHT > len(ids):
                continue
            key = hashlib.sha256(
                f"odd8h2-sparse-graph-v1|skip39000|{source_row}|{position}|{token}".encode()
            ).hexdigest()
            candidates.append((key, source_row, position, token))
    selected, used = [], set()
    for item in sorted(candidates):
        if item[1] in used:
            continue
        selected.append(item)
        used.add(item[1])
        if len(selected) == CONTEXTS:
            break
    assert len(selected) == CONTEXTS

    contexts, rows = [], []
    for context_id, (_, source_row, position, original_token) in enumerate(selected):
        chunk = source[source_row, position - LEFT:position + RIGHT].tolist()
        assert len(chunk) == LEFT + RIGHT and chunk[LEFT] == original_token
        counterpart = token_pair[original_token]
        variants = {}
        for token in (original_token, counterpart):
            ids = list(chunk)
            ids[LEFT] = token
            variants[region[token]] = ids
        contexts.append({
            "context_id": context_id,
            "source_cache": "fineweb_n192_skip39000.pt",
            "source_row": source_row,
            "source_city_position": position,
            "city_position": LEFT,
            "original_city": label[original_token],
            "counterpart_city": label[counterpart],
            "original_region": region[original_token],
            "natural_text": enc.decode(chunk),
            "natural_ids_sha256": hashlib.sha256(
                torch.tensor(chunk, dtype=torch.int64).numpy().tobytes()
            ).hexdigest(),
        })
        for endpoint, (uk_id, us_id) in enumerate(endpoints):
            for cue in ("British", "American"):
                token = original_token if cue == region[original_token] else counterpart
                record = {
                    "row_id": len(rows),
                    "context_id": context_id,
                    "panel": "fresh_natural_ood",
                    "source_cache": "fineweb_n192_skip39000.pt",
                    "source_row": source_row,
                    "cue": cue,
                    "is_untouched_natural_arm": cue == region[original_token],
                    "city": label[token],
                    "city_position": LEFT,
                    "destination_positions": list(range(LEFT + 1, len(chunk) - 1)),
                    "ids": variants[cue],
                    "uk_id": uk_id,
                    "us_id": us_id,
                    "endpoint": endpoint,
                }
                payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                record["row_sha256"] = hashlib.sha256(payload).hexdigest()
                rows.append(record)

    paired = all(
        rows[index]["cue"] == "British"
        and rows[index + 1]["cue"] == "American"
        and rows[index]["context_id"] == rows[index + 1]["context_id"]
        and rows[index]["endpoint"] == rows[index + 1]["endpoint"]
        and sum(a != b for a, b in zip(rows[index]["ids"], rows[index + 1]["ids"])) == 1
        for index in range(0, len(rows), 2)
    )
    output = {
        "schema": "odd_attention8h2.sparse_graph.rows.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "selection": (
            "SHA256-ordered city occurrences from the previously unused skip39000 cache; "
            "one occurrence per source row; no model execution, activations, logits, or endpoint occurrence."
        ),
        "source_cache": str(SOURCE),
        "city_pairs": CITY_PAIRS,
        "endpoints": ENDPOINTS,
        "window": {"left_of_city": LEFT, "city": 1, "right_from_city": RIGHT, "tokens": LEFT + RIGHT},
        "contexts": contexts,
        "rows": rows,
    }
    OUT_ROWS.write_text(json.dumps(output, indent=2) + "\n")
    control = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pred_a": paired and len(rows) == 96,
        "candidate_occurrences": len(candidates),
        "selected_contexts": len(contexts),
        "rows": len(rows),
        "pairs": len(rows) // 2,
        "unique_source_rows": len({item["source_row"] for item in contexts}),
        "source_sha256": digest(SOURCE),
        "rows_sha256": digest(OUT_ROWS),
        "scope": (
            "Outcome-blind natural FineWeb fragments from skip39000 with one untouched arm and one "
            "single-token city substitution; no capability or causal claim."
        ),
    }
    OUT_CONTROL.write_text(json.dumps(control, indent=2) + "\n")
    print(json.dumps(control, indent=2))


if __name__ == "__main__":
    main()
