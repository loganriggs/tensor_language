"""Freeze a second, source-disjoint FineWeb panel for typed-face replication."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
import torch


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt"
PRIOR = HERE / "ODD_ATTENTION8H2_CORPUS_TRANSFER_V1_ROWS.json"
OUT_ROWS = HERE / "TYPED_FACE_KEY_SOURCE_FRESH_V1_ROWS.json"
OUT_CONTROL = HERE / "TYPED_FACE_KEY_SOURCE_FRESH_V1_CPU_CONTROL.json"
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
LEFT, RIGHT, CONTEXTS = 8, 24, 8


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if OUT_ROWS.exists() or OUT_CONTROL.exists():
        raise FileExistsError("typed-face replication row outputs already exist")
    encoder = tiktoken.get_encoding("gpt2")
    token_pair, region, label = {}, {}, {}
    for uk, us in CITY_PAIRS:
        uk_ids, us_ids = encoder.encode(uk), encoder.encode(us)
        assert len(uk_ids) == len(us_ids) == 1
        ui, si = uk_ids[0], us_ids[0]
        token_pair[ui], token_pair[si] = si, ui
        region[ui], region[si] = "British", "American"
        label[ui], label[si] = uk.strip(), us.strip()
    endpoints = []
    for uk, us in ENDPOINTS:
        uk_ids, us_ids = encoder.encode(uk), encoder.encode(us)
        assert len(uk_ids) == len(us_ids) == 1
        endpoints.append((uk_ids[0], us_ids[0]))

    excluded = {item["source_row"] for item in json.loads(PRIOR.read_text())["contexts"] if item["corpus"] == "fineweb"}
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    candidates = []
    for source_row, tensor_row in enumerate(source):
        if source_row in excluded:
            continue
        for position, token in enumerate(tensor_row.tolist()):
            if token not in token_pair or position < LEFT or position + RIGHT > len(tensor_row):
                continue
            key = hashlib.sha256(
                f"typed-face-key-source-fresh-v1|{source_row}|{position}|{token}".encode()
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
            "source_cache": "fineweb_n192_skip11000.pt",
            "source_row": source_row,
            "source_city_position": position,
            "city_position": LEFT,
            "original_city": label[original_token],
            "counterpart_city": label[counterpart],
            "original_region": region[original_token],
            "natural_text": encoder.decode(chunk),
            "natural_ids_sha256": hashlib.sha256(
                torch.tensor(chunk, dtype=torch.int64).numpy().tobytes()
            ).hexdigest(),
        })
        for endpoint, (uk_id, us_id) in enumerate(endpoints):
            for cue in ("British", "American"):
                token = original_token if cue == region[original_token] else counterpart
                record = {
                    "row_id": len(rows), "context_id": context_id,
                    "panel": "key_source_fresh_shifted_position",
                    "source_cache": "fineweb_n192_skip11000.pt", "source_row": source_row,
                    "cue": cue, "is_untouched_natural_arm": cue == region[original_token],
                    "city": label[token], "city_position": LEFT,
                    "destination_positions": list(range(LEFT + 1, len(chunk) - 1)),
                    "ids": variants[cue], "uk_id": uk_id, "us_id": us_id,
                    "endpoint": endpoint,
                }
                payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                record["row_sha256"] = hashlib.sha256(payload).hexdigest()
                rows.append(record)
    paired = all(
        rows[index]["cue"] == "British" and rows[index + 1]["cue"] == "American"
        and rows[index]["context_id"] == rows[index + 1]["context_id"]
        and rows[index]["endpoint"] == rows[index + 1]["endpoint"]
        and sum(a != b for a, b in zip(rows[index]["ids"], rows[index + 1]["ids"])) == 1
        for index in range(0, len(rows), 2)
    )
    output = {
        "schema": "typed_face.key_source_fresh.rows.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "selection": (
            "SHA256-ordered eligible occurrences after excluding every V1 natural-panel source row; "
            "one occurrence per source row; no model execution, activations, logits, or endpoint occurrence."
        ),
        "source_cache": str(SOURCE), "prior_rows_sha256": digest(PRIOR),
        "excluded_prior_source_rows": sorted(excluded),
        "city_pairs": CITY_PAIRS, "endpoints": ENDPOINTS,
        "window": {"left_of_city": LEFT, "city": 1, "right_from_city": RIGHT, "tokens": LEFT + RIGHT},
        "contexts": contexts, "rows": rows,
    }
    OUT_ROWS.write_text(json.dumps(output, indent=2) + "\n")
    control = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "pred_a": paired and len(rows) == 96 and not (excluded & used),
        "candidate_occurrences_after_exclusion": len(candidates),
        "selected_contexts": len(contexts), "rows": len(rows), "pairs": len(rows) // 2,
        "unique_source_rows": len(used), "prior_source_overlap": len(excluded & used),
        "source_sha256": digest(SOURCE), "prior_rows_sha256": digest(PRIOR),
        "rows_sha256": digest(OUT_ROWS),
        "scope": (
            "Outcome-blind shifted-position FineWeb panel excluding earlier regional source documents within skip11000; "
            "no native capability, logits, or causal result."
        ),
    }
    OUT_CONTROL.write_text(json.dumps(control, indent=2) + "\n")
    print(json.dumps(control, indent=2))


if __name__ == "__main__":
    main()
