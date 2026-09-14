"""Combine four frozen constructions for head8.2 destination-role testing."""
import hashlib
import json
from pathlib import Path

from odd_framing_role_split_v1 import role_masks
from regional_cue_row_check_v1 import validate


def main():
    root = Path(__file__).resolve().parent
    out = root / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ROWS.json"
    if out.exists():
        raise FileExistsError(out)
    sources = ["ODD_FRAMING_FRESH_V1_ROWS.json", "ODD_ATTENTION8H2_CHAIN_FRESH_V1_ROWS.json"]
    rows = []
    templates = []
    for group, name in enumerate(sources):
        data = json.loads((root / name).read_text())
        templates.extend(data["templates"])
        for row in data["rows"]:
            item = dict(row)
            item["family"] = group * 2 + row["family"]
            item["template"] = item["family"]
            item["parent_row_id"] = item.pop("row_id")
            item["row_id"] = hashlib.sha256(json.dumps(item, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            rows.append(item)
    assert len(rows) == 96
    validate(rows)
    masks = role_masks(rows)
    counts = [{key: int(value.sum()) for key, value in mask.items()} for mask in masks]
    result = {
        "schema": "odd_attention8h2_destination_role.rows.v1",
        "selection": "Exact union of two previously frozen 48-row panels; family labels offset before any destination-role model scores.",
        "sources": sources, "templates": templates, "representative_counts": [counts[index] for index in (0, 24, 48, 72)],
        "rows": rows,
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "families": sorted(set(row["family"] for row in rows)), "representative_counts": result["representative_counts"]}))


if __name__ == "__main__":
    main()
