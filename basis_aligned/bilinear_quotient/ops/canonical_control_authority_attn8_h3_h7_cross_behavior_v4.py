#!/usr/bin/env python3
# BQLANE: cpu
"""Frozen full-family R567 control pairing for cross-behavior factor v4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
RECEIPT = ROOT / "increment_two_hypothesis_rows_rung567_receipt.json"
CAPABILITY = ROOT / "numeric_two_hypothesis_capability_rung569_570_results.json"
STEP_TWO = ROOT / "numbered_list_conflict_confirmation_rung572_results.json"
CPU_AUDIT = ROOT / "numeric_two_hypothesis_capability_rung571_audit.json"
LIST_MAP = ROOT / "numbered_list_semantic_positions_rung573.json"
NUMERIC_MAP = ROOT / "numeric_sequence_semantic_positions_rung577.json"
HASHES = {ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
          RECEIPT: "02b2c37cc23434138accd63e920f417cda10f1c86a4c08c174537149ec2b1072",
          CAPABILITY: "7cc56f22def334673e0035fad7c6a7d1fc58ab8edd3a99744bebd9fb4e6af7e7",
          STEP_TWO: "3df046bdcc4fa4387a2dbef084ed732c5f6a05232b7fa64072af3cd4939daea1",
          CPU_AUDIT: "c5453ddaa4aa46806cbfcb9a9b0941fe8ddbb21c61e5e22d00c1d1cea6dd74bb",
          LIST_MAP: "b4f4f8e58c03deb3a015141656572dcf1fd0fe0c12027c358b441459c245c16b",
          NUMERIC_MAP: "a6a98715617cf91971655c252553f42d45b59937ecfbf46722b518333721de1d"}
DECISIVE = ("list_step_two_conflict", "sequence_digit_copy_control",
            "sequence_word_copy_control")
SECONDARY = "list_repeated_index_control"
FAMILIES = DECISIVE + (SECONDARY,)
ROW_ID_DIGESTS = {
 ("list_step_two_conflict", "FIT"): "089f5aaaf0f21c6499d04f0132707b87b20a14abf5fd54c9e246f10626836e22",
 ("list_step_two_conflict", "SELECT"): "beba193ebbc6b0e091f0d910daafa33597ede6afcd314b84463805f4f515a7fe",
 ("list_repeated_index_control", "FIT"): "a136454f307b808cbb7b3c9cb6d6d40ad499e078cec2e32102add52c1050a75e",
 ("list_repeated_index_control", "SELECT"): "c24323974a1bf9ae2478cb65db56c1e7a1d7af144d8150ce75df7adc7dbd3e6e",
 ("sequence_digit_copy_control", "FIT"): "50496a54b1c1c9c8e89049f3330178016226947d512865be049f8bb7545db391",
 ("sequence_digit_copy_control", "SELECT"): "d546a1d6db97cf9360313ef5913e67bdfe58259e08a7851a22eab797454482fa",
 ("sequence_word_copy_control", "FIT"): "22984d24d958ba3123630178df9a93609f6c2ec9a0ecf165e20d635b72281e4b",
 ("sequence_word_copy_control", "SELECT"): "4f8bf47e20efa3f0c150e16dae5dd3576794e9fafc7c564014f60843928d33e1"}
EXPECTED_PAIRING_SHA256 = "afcf3d6adb3a2b47635129ec22447948789ecbf144575143bb319dc851dcfc9f"


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value): return hashlib.sha256(json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _endpoint(row, mapping, side):
    item = mapping["endpoints"][side]
    positions = item.get("label_positions") or item.get("value_positions")
    final = positions[-1]
    value = final.get("label", final.get("numeric_value"))
    return {"ids": row[f"{side}_ids"], "text": row[f"{side}_text"],
            "answer_id": row[f"{side}_answer_id"], "answer_text": row[f"{side}_answer"],
            "source_positions": [entry["token_position"] for entry in positions],
            "query_position": item.get("final_query_position", item.get("query_position")),
            "final_value": value}


def build_pairs(verify_frozen=True):
    for path, expected in HASHES.items():
        if sha256(path) != expected: raise RuntimeError(f"frozen input changed: {path}")
    rows = json.loads(ROWS.read_text())["rows"]
    list_maps = {item["row_id"]: item for item in json.loads(LIST_MAP.read_text())["mappings"]}
    numeric_maps = {item["row_id"]: item for item in json.loads(NUMERIC_MAP.read_text())["records"]}
    output = []
    for family in FAMILIES:
        maps = list_maps if family.startswith("list_") else numeric_maps
        for split in ("FIT", "SELECT"):
            selected = [row for row in rows if row["family_id"] == family and row["split"] == split]
            ids_digest = hashlib.sha256(json.dumps(
                [row["row_id"] for row in selected], separators=(",", ":")).encode()).hexdigest()
            if ids_digest != ROW_ID_DIGESTS[(family, split)]:
                raise RuntimeError(f"full-family row order changed: {family} {split}")
            by_value = {}
            for row in selected:
                value = _endpoint(row, maps[row["row_id"]], "base")["final_value"]
                by_value.setdefault(value, []).append(row)
            values = sorted(by_value)
            if len(values) != 2 or values[1] != values[0]+1 \
                    or len(by_value[values[0]]) != len(by_value[values[1]]):
                raise RuntimeError(f"adjacent complete value sets changed: {family} {split}")
            lower = sorted(by_value[values[0]], key=lambda row: row["row_id"])
            higher = sorted(by_value[values[1]], key=lambda row: row["row_id"])
            for pair_index, (low, high) in enumerate(zip(lower, higher)):
                for side in ("base", "donor"):
                    low_ep = _endpoint(low, maps[low["row_id"]], side)
                    high_ep = _endpoint(high, maps[high["row_id"]], side)
                    if len(low_ep["source_positions"]) != 3 \
                            or len(high_ep["source_positions"]) != 3 \
                            or low_ep["query_position"] != len(low_ep["ids"])-1 \
                            or high_ep["query_position"] != len(high_ep["ids"])-1 \
                            or high_ep["final_value"] != low_ep["final_value"]+1:
                        raise RuntimeError("paired source maps or adjacent semantics changed")
                    for direction, recipient, donor in (
                        ("lower_to_higher", low_ep, high_ep),
                        ("higher_to_lower", high_ep, low_ep)):
                        output.append({"pair_id": canonical([family, split, pair_index, side, direction]),
                            "family_id": family, "decisive": family in DECISIVE,
                            "split": split, "surface_side": side, "direction": direction,
                            "lower_row_id": low["row_id"], "higher_row_id": high["row_id"],
                            "recipient": recipient, "donor": donor})
    digest = canonical(output)
    if verify_frozen and digest != EXPECTED_PAIRING_SHA256:
        raise RuntimeError(f"canonical control pairing changed: {digest}")
    return output


def compile_plan():
    pairs = build_pairs()
    return {"schema": "attn8_h3_h7_cross_behavior_v4_canonical_control_authority",
            "pair_count": len(pairs), "pairing_sha256": canonical(pairs),
            "decisive_families": list(DECISIVE), "secondary_family": SECONDARY,
            "model_loaded": False, "outcomes_opened": []}


if __name__ == "__main__": print(json.dumps(compile_plan(), indent=2, sort_keys=True))
