"""Freeze a score-blind natural-corpus transfer panel for the head8.2 -> O edge."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import tiktoken
import torch


HERE = Path(__file__).resolve().parent
FINEWEB = HERE.parent / "bilinear_quotient/.rowcache/fineweb_n192_skip11000.pt"
PILE = HERE / "SHARED_NODE_PARENT1_CORPUS_SHIFT_V1_PREFIXES.pt"
PILE_RECEIPT = HERE / "SHARED_NODE_PARENT1_CORPUS_SHIFT_V1_ROWS.json"
OUT_ROWS = HERE / "ODD_ATTENTION8H2_CORPUS_TRANSFER_V1_ROWS.json"
OUT_CONTROL = HERE / "ODD_ATTENTION8H2_CORPUS_TRANSFER_V1_CPU_CONTROL.json"

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
LEFT, RIGHT = 12, 20
DOMAIN_LIMITS = {"fineweb": 8, "pile": 4}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    if OUT_ROWS.exists() or OUT_CONTROL.exists():
        raise FileExistsError("corpus-transfer output already exists")
    enc = tiktoken.get_encoding("gpt2")
    token_pair = {}
    region = {}
    label = {}
    for uk, us in CITY_PAIRS:
        ui, si = enc.encode(uk), enc.encode(us)
        assert len(ui) == len(si) == 1
        token_pair[ui[0]], token_pair[si[0]] = si[0], ui[0]
        region[ui[0]], region[si[0]] = "British", "American"
        label[ui[0]], label[si[0]] = uk.strip(), us.strip()
    endpoint_ids = []
    for uk, us in ENDPOINTS:
        ui, si = enc.encode(uk), enc.encode(us)
        assert len(ui) == len(si) == 1
        endpoint_ids.append((ui[0], si[0]))

    pile_meta_raw = json.loads(PILE_RECEIPT.read_text())["metadata"]
    pile_meta = {}
    for item in pile_meta_raw:
        pile_meta.setdefault(item["source_row"], item)
    sources = {
        "fineweb": torch.load(FINEWEB, map_location="cpu", weights_only=True),
        "pile": torch.load(PILE, map_location="cpu", weights_only=True),
    }
    selected = {}
    candidate_counts = {}
    for corpus, tensor in sources.items():
        candidates = []
        for source_row, row in enumerate(tensor):
            ids = row.tolist()
            for pos, tok in enumerate(ids):
                if tok not in token_pair or pos < LEFT or pos + RIGHT > len(ids):
                    continue
                key = hashlib.sha256(f"odd8h2-corpus-v1|{corpus}|{source_row}|{pos}|{tok}".encode()).hexdigest()
                candidates.append((key, source_row, pos, tok))
        candidate_counts[corpus] = len(candidates)
        # At most one occurrence per source document; selection uses no model output.
        chosen, used = [], set()
        for item in sorted(candidates):
            if item[1] in used:
                continue
            chosen.append(item)
            used.add(item[1])
            if len(chosen) == DOMAIN_LIMITS[corpus]:
                break
        assert len(chosen) == DOMAIN_LIMITS[corpus]
        selected[corpus] = chosen

    contexts, rows = [], []
    context_id = 0
    for corpus in ("fineweb", "pile"):
        tensor = sources[corpus]
        for _, source_row, pos, original_tok in selected[corpus]:
            chunk = tensor[source_row, pos - LEFT:pos + RIGHT].tolist()
            assert len(chunk) == LEFT + RIGHT and chunk[LEFT] == original_tok
            counterpart = token_pair[original_tok]
            variants = {}
            for tok in (original_tok, counterpart):
                ids = list(chunk)
                ids[LEFT] = tok
                variants[region[tok]] = ids
            meta = pile_meta.get(source_row, {}) if corpus == "pile" else {}
            ctx = {
                "context_id": context_id, "corpus": corpus,
                "pile_domain": meta.get("domain"), "pile_subset": meta.get("subset"),
                "source_row": source_row, "source_city_position": pos,
                "city_position": LEFT, "original_city": label[original_tok],
                "counterpart_city": label[counterpart],
                "original_region": region[original_tok],
                "natural_text": enc.decode(chunk),
                "natural_ids_sha256": hashlib.sha256(torch.tensor(chunk, dtype=torch.int64).numpy().tobytes()).hexdigest(),
            }
            contexts.append(ctx)
            for endpoint, (uk_id, us_id) in enumerate(endpoint_ids):
                for cue in ("British", "American"):
                    ids = variants[cue]
                    record = {
                        "row_id": len(rows), "context_id": context_id,
                        "corpus": corpus, "pile_domain": meta.get("domain"),
                        "source_row": source_row, "cue": cue,
                        "is_untouched_natural_arm": cue == region[original_tok],
                        "city": label[original_tok if cue == region[original_tok] else counterpart],
                        "city_position": LEFT,
                        "destination_positions": list(range(LEFT + 1, len(ids) - 1)),
                        "ids": ids, "uk_id": uk_id, "us_id": us_id,
                        "control_ids": [670, 3946], "endpoint": endpoint,
                    }
                    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
                    record["row_sha256"] = hashlib.sha256(payload).hexdigest()
                    rows.append(record)
            context_id += 1

    paired = all(
        rows[i]["context_id"] == rows[i + 1]["context_id"]
        and rows[i]["endpoint"] == rows[i + 1]["endpoint"]
        and sum(a != b for a, b in zip(rows[i]["ids"], rows[i + 1]["ids"])) == 1
        and rows[i]["city_position"] == rows[i + 1]["city_position"] == LEFT
        for i in range(0, len(rows), 2)
    )
    output = {
        "schema": "odd_attention8h2.corpus_transfer.rows.v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Hash-ordered one occurrence per cached source row; city occurrence and fixed window only; no model execution, logits, activations, or endpoint occurrence used.",
        "city_pairs": CITY_PAIRS, "endpoints": ENDPOINTS,
        "window": {"left_of_city": LEFT, "city": 1, "right_from_city": RIGHT, "tokens": LEFT + RIGHT},
        "contexts": contexts, "rows": rows,
    }
    OUT_ROWS.write_text(json.dumps(output, indent=2) + "\n")
    control = {
        "utc": datetime.now(timezone.utc).isoformat(), "pred_a": paired,
        "candidate_occurrences": candidate_counts,
        "selected_contexts": {k: len(v) for k, v in selected.items()},
        "rows": len(rows), "pairs": len(rows) // 2,
        "pile_domains": sorted({x["pile_domain"] for x in contexts if x["corpus"] == "pile"}),
        "source_sha256": {str(FINEWEB): digest(FINEWEB), str(PILE): digest(PILE), str(PILE_RECEIPT): digest(PILE_RECEIPT)},
        "rows_sha256": digest(OUT_ROWS),
        "scope": "Outcome-blind cached natural-text fragments with one untouched arm and one single-token cross-region city substitution. No native capability, score selection, pretraining-disjointness, or causal result.",
    }
    OUT_CONTROL.write_text(json.dumps(control, indent=2) + "\n")
    print(json.dumps(control))


if __name__ == "__main__":
    main()
