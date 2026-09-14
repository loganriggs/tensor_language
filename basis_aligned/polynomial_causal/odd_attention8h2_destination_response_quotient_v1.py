"""Outcome-blind context deduplication for the destination response quotient."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from odd_contextual_positions_v1 import contextual_masks
from odd_framing_role_split_v1 import role_masks


def main():
    root = Path(__file__).resolve().parent
    out_rows = root / "ODD_ATTENTION8H2_DESTINATION_RESPONSE_QUOTIENT_V1_ROWS.json"
    out_control = root / "ODD_ATTENTION8H2_DESTINATION_RESPONSE_QUOTIENT_V1_CPU_CONTROL.json"
    if out_rows.exists() or out_control.exists():
        raise FileExistsError("destination quotient output exists")
    parent = json.loads((root / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ROWS.json").read_text())["rows"]
    ctx_masks, role = contextual_masks(parent), role_masks(parent)
    grouped = {}
    for i, row in enumerate(parent):
        key = (row["family"], row["pair"], row["cue"])
        cell = grouped.setdefault(key, {"indices": [], "endpoint_ids": {}})
        cell["indices"].append(i)
        cell["endpoint_ids"][row["endpoint"]] = [row["uk_id"], row["us_id"]]
    contexts = []
    for key in sorted(grouped):
        family, pair, cue = key
        cell = grouped[key]
        first = parent[cell["indices"][0]]
        assert len(cell["indices"]) == 6 and sorted(cell["endpoint_ids"]) == list(range(6))
        assert all(parent[i]["ids"] == first["ids"] for i in cell["indices"])
        assert all(role[i]["framing"].equal(role[cell["indices"][0]]["framing"]) for i in cell["indices"])
        framing = role[cell["indices"][0]]["framing"].nonzero().flatten().tolist()
        city = ctx_masks[cell["indices"][0]]["city"].nonzero().flatten().tolist()
        assert len(city) == 1 and framing
        item = {
            "context_id": len(contexts), "family": family, "split": "discovery" if family < 2 else "heldout",
            "pair": pair, "cue": cue, "city": first["city"], "ids": first["ids"],
            "city_position": city[0], "destination_positions": framing,
            "endpoint_ids": [cell["endpoint_ids"][j] for j in range(6)],
            "control_ids": [[3797, 3290], [2266, 4171], [3321, 3431], [17180, 10912]],
        }
        payload = json.dumps(item, sort_keys=True, separators=(",", ":")).encode()
        item["row_sha256"] = hashlib.sha256(payload).hexdigest()
        contexts.append(item)
    paired = all(
        contexts[i]["family"] == contexts[i + 1]["family"]
        and contexts[i]["pair"] == contexts[i + 1]["pair"]
        and contexts[i]["cue"] == "American" and contexts[i + 1]["cue"] == "British"
        and contexts[i]["destination_positions"] == contexts[i + 1]["destination_positions"]
        and sum(a != b for a, b in zip(contexts[i]["ids"], contexts[i + 1]["ids"])) == 1
        for i in range(0, len(contexts), 2)
    )
    doc = {
        "schema": "odd_attention8h2.destination_response_quotient.rows.v1",
        "selection": "Exact deduplication of the frozen four-family destination-role panel before new native scores; families 0-1 discovery, 2-3 held out.",
        "contexts": contexts,
    }
    out_rows.write_text(json.dumps(doc, indent=2) + "\n")
    receipt = {
        "utc": datetime.now(timezone.utc).isoformat(), "pred_a": paired,
        "contexts": len(contexts), "discovery_contexts": sum(x["split"] == "discovery" for x in contexts),
        "heldout_contexts": sum(x["split"] == "heldout" for x in contexts),
        "destination_count_range": [min(len(x["destination_positions"]) for x in contexts), max(len(x["destination_positions"]) for x in contexts)],
        "rows_sha256": hashlib.sha256(out_rows.read_bytes()).hexdigest(),
        "scope": "Outcome-blind deduplication and exact masks only; no response rank, fit, native score, or circuit verdict.",
    }
    out_control.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
