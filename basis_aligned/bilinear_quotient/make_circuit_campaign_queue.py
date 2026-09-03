#!/usr/bin/env python3
"""Generate the auditable work queue for behavior circuits and shared subroutines."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from make_circuit_coverage import CATEGORIES, active_events, category_cell


ROOT = Path(__file__).resolve().parent
CIRCUITS = ROOT / "circuits"
REGISTRY = CIRCUITS / "registry.json"
OUT_JSON = CIRCUITS / "campaign_queue.json"
OUT_MD = ROOT / "CIRCUIT_CAMPAIGN_QUEUE.md"
BLOCKING = {"failed", "null", "invalid"}


def build() -> dict:
    compact = json.loads(REGISTRY.read_text())["circuits"]
    work = []
    legacy = []
    seen_events: set[str] = set()
    for tag, row in sorted(compact.items()):
        path = CIRCUITS / row["file"]
        document = json.loads(path.read_text())
        if row.get("schema_version") != 2:
            legacy.append({
                "tag": tag,
                "record_path": str(path.relative_to(ROOT)),
                "status": row.get("status", "legacy"),
                "action": "candidate_only_do_not_count_as_counterfactual_circuit",
            })
            continue
        active_id = row["active_claim_id"]
        claim = next(item for item in document["claims"] if item["claim_id"] == active_id)
        events = active_events(document)
        event_ids = [event["event_id"] for event in events]
        overlap = seen_events.intersection(event_ids)
        if overlap:
            raise RuntimeError(f"events occur in more than one record: {sorted(overlap)}")
        seen_events.update(event_ids)
        family_status = Counter(family["status"] for family in claim["counterfactual_families"])
        open_events = [
            event["event_id"] for event in events
            if event["stage"] == "preregistered" and event["verdict"] == "inconclusive"
        ]
        held_events = [event["event_id"] for event in events if event["verdict"] == "held"]
        blocking_events = [event["event_id"] for event in events if event["verdict"] in BLOCKING]
        coverage = {name: category_cell(events, types) for name, types in CATEGORIES.items()}
        gaps = [name for name, cell in coverage.items() if cell["status"] in {"missing", "blocked", "mixed"}]
        if open_events:
            work_state = "experiment_open"
        elif claim["status"] in {"validated", "compiled"} and not gaps:
            work_state = "acceptance_complete"
        elif any(family["status"] == "proposed" for family in claim["counterfactual_families"]):
            work_state = "dataset_or_family_definition_needed"
        else:
            work_state = "next_experiment_needed"
        work.append({
            "tag": tag,
            "kind": row["kind"],
            "record_path": str(path.relative_to(ROOT)),
            "active_claim_id": active_id,
            "claim_status": claim["status"],
            "causal_variable_id": claim["causal_variable"]["id"],
            "counterfactual_family_count": len(claim["counterfactual_families"]),
            "counterfactual_family_status_counts": dict(sorted(family_status.items())),
            "split_plan_ids": claim["split_plan_ids"],
            "active_event_ids": event_ids,
            "open_event_ids": open_events,
            "held_event_ids": held_events,
            "blocking_event_ids": blocking_events,
            "coverage": {name: cell["status"] for name, cell in coverage.items()},
            "acceptance_gaps": gaps,
            "work_state": work_state,
            "next_missing": claim["next_missing"],
        })
    order = {
        "experiment_open": 0,
        "next_experiment_needed": 1,
        "dataset_or_family_definition_needed": 2,
        "acceptance_complete": 3,
    }
    work.sort(key=lambda item: (order[item["work_state"]], item["tag"]))
    return {
        "schema": "circuit_campaign_queue_v1",
        "definitions": {
            "canonical_work_item": "one task-defined behavior circuit or shared subroutine with a v2 record",
            "legacy_candidate": "a census slice; not a counterfactually established circuit",
            "experiment_open": "a preregistered event has no superseding completed result",
            "acceptance_gap": "missing, blocked, or mixed evidence in one program acceptance category",
        },
        "summary": {
            "canonical_work_items": len(work),
            "legacy_candidates": len(legacy),
            "open_experiments": sum(len(item["open_event_ids"]) for item in work),
            "held_events": sum(len(item["held_event_ids"]) for item in work),
            "blocking_events": sum(len(item["blocking_event_ids"]) for item in work),
        },
        "work_items": work,
        "legacy_candidates": legacy,
    }


def clean(value: object) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def main() -> None:
    payload = build()
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")
    summary = payload["summary"]
    lines = [
        "# Circuit campaign queue (generated by make_circuit_campaign_queue.py — do not edit)",
        "",
        "This is the anti-duplication work view. It distinguishes task-defined circuits from legacy census slices, "
        "which are candidates rather than counterfactually established circuits. Open the linked canonical JSON "
        "record before changing a dataset, intervention, or verdict.",
        "",
        f"{summary['canonical_work_items']} canonical work items; {summary['legacy_candidates']} legacy candidates; "
        f"{summary['open_experiments']} open experiments; {summary['held_events']} active held events; "
        f"{summary['blocking_events']} active blocking events.",
        "",
        "| circuit | state | claim | counterfactual families | open experiments | held / blocking | gaps | exact next work | record |",
        "|---|---|---|---:|---|---:|---|---|---|",
    ]
    for item in payload["work_items"]:
        family_counts = ", ".join(f"{key}:{value}" for key, value in item["counterfactual_family_status_counts"].items())
        lines.append(
            f"| `{item['tag']}` | {item['work_state']} | `{item['active_claim_id']}` ({item['claim_status']}) | "
            f"{item['counterfactual_family_count']} ({family_counts}) | "
            f"{clean(', '.join(item['open_event_ids']) or '—')} | "
            f"{len(item['held_event_ids'])} / {len(item['blocking_event_ids'])} | "
            f"{clean(', '.join(item['acceptance_gaps']) or '—')} | {clean(item['next_missing'])} | "
            f"[{item['record_path']}]({item['record_path']}) |"
        )
    lines.extend([
        "",
        "## Scale accounting",
        "",
        f"The {summary['legacy_candidates']} legacy census slices remain searchable in `CIRCUITS_INDEX.md`, but they "
        "do not enter the evidential-circuit count until they receive a task-defined causal variable, multiple valid "
        "counterfactual families, held-out splits, causal interventions, selectivity controls, and a canonical v2 record.",
    ])
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {OUT_MD.name} and {OUT_JSON.relative_to(ROOT)}: "
        f"{summary['canonical_work_items']} work items, {summary['open_experiments']} open experiments"
    )


if __name__ == "__main__":
    main()
