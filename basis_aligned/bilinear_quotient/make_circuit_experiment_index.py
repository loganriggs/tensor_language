#!/usr/bin/env python3
"""Generate a human-readable experiment ledger and duplicate-work audit for v2 circuits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CIRCUITS = ROOT / "circuits"
REGISTRY = CIRCUITS / "registry.json"
OUT_JSON = CIRCUITS / "experiment_index.json"
OUT_MD = ROOT / "CIRCUIT_EXPERIMENT_INDEX.md"


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def protocol_key(tag: str, record: dict, event: dict) -> str:
    """Hash scientific design while ignoring claim revision, result, seed, and split execution."""
    claim = next(item for item in record["claims"] if item["claim_id"] == event["claim_id"])
    families = {
        family["family_id"]: {
            "role": family["role"],
            "changes": family["changes"],
            "holds_fixed": family["holds_fixed"],
            "control_ids": family["control_ids"],
        }
        for family in claim["counterfactual_families"]
        if family["family_id"] in event.get("family_ids", [])
    }
    return canonical_hash({
        "tag": tag,
        "causal_variable_id": claim["causal_variable"]["id"],
        "families": families,
        "site_id": event.get("site_id"),
        "test_type": event["test_type"],
        "metric_contract": [
            {key: metric.get(key) for key in ("name", "bar")} for metric in event.get("metrics", [])
        ],
    })


def load_events() -> list[dict]:
    compact = json.loads(REGISTRY.read_text())["circuits"]
    rows = []
    for tag, registry_row in sorted(compact.items()):
        if registry_row.get("schema_version") != 2:
            continue
        record = json.loads((CIRCUITS / registry_row["file"]).read_text())
        for event in record.get("evidence_events", []):
            rows.append({
                "circuit": tag,
                "event_id": event["event_id"],
                "claim_id": event["claim_id"],
                "test_type": event["test_type"],
                "stage": event["stage"],
                "verdict": event["verdict"],
                "failure_kind": event.get("failure_kind"),
                "design_key": event["design_key"],
                "protocol_key": protocol_key(tag, record, event),
                "execution_key": event["execution_key"],
                "supersedes_event_id": event.get("supersedes_event_id"),
                "replicates_event_id": event.get("replicates_event_id"),
                "prereg_artifact_id": event.get("prereg_artifact_id"),
                "result_artifact_id": event.get("result_artifact_id"),
            })
    return rows


def audit(events: list[dict]) -> dict:
    by_id = {row["event_id"]: row for row in events}
    if len(by_id) != len(events):
        raise RuntimeError("event IDs are not globally unique")
    superseded = {row["supersedes_event_id"] for row in events if row["supersedes_event_id"]}
    open_preregistrations = [
        row for row in events
        if row["stage"] == "preregistered" and row["verdict"] == "inconclusive"
        and row["event_id"] not in superseded
    ]

    executions = defaultdict(list)
    protocols = defaultdict(list)
    designs = defaultdict(list)
    for row in events:
        executions[row["execution_key"]].append(row)
        protocols[row["protocol_key"]].append(row)
        designs[row["design_key"]].append(row)
    duplicate_executions = [group for group in executions.values() if len(group) > 1]

    review = []
    for key, group in protocols.items():
        completed = [row for row in group if row["stage"] in {"complete", "invalid"}]
        planned = [row for row in group if row in open_preregistrations]
        unlinked_completed = [
            row for row in completed
            if len(completed) > 1 and not row["supersedes_event_id"] and not row["replicates_event_id"]
        ]
        planned_duplicates = []
        if completed:
            completed_ids = {row["event_id"] for row in completed}
            planned_duplicates = [
                row for row in planned
                if row["supersedes_event_id"] not in completed_ids
                and row["replicates_event_id"] not in completed_ids
            ]
        if unlinked_completed or planned_duplicates:
            review.append({
                "protocol_key": key,
                "event_ids": [row["event_id"] for row in group],
                "unlinked_completed_event_ids": [row["event_id"] for row in unlinked_completed],
                "planned_duplicate_event_ids": [row["event_id"] for row in planned_duplicates],
            })

    return {
        "event_count": len(events),
        "design_count": len(designs),
        "protocol_count": len(protocols),
        "execution_count": len(executions),
        "open_preregistrations": open_preregistrations,
        "duplicate_execution_groups": [[row["event_id"] for row in group] for group in duplicate_executions],
        "protocol_repeat_review": review,
    }


def cell(value: object) -> str:
    return str(value if value is not None else "—").replace("|", "/")


def main() -> None:
    events = load_events()
    report = audit(events)
    payload = {
        "schema": "circuit_experiment_index_v1",
        "definitions": {
            "design_key": "registered design including claim revision",
            "protocol_key": "scientific design ignoring claim revision and execution-specific split/seed/result",
            "execution_key": "design plus split, seed, checkpoint, and bound artifact hashes",
            "open_preregistration": "preregistered inconclusive event not superseded by a result",
            "protocol_repeat_review": "repeat lacking an explicit supersedes or replicates link; inspect before running",
        },
        **report,
        "events": events,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=1) + "\n")

    lines = [
        "# Circuit experiment index (generated by make_circuit_experiment_index.py — do not edit)", "",
        "Check this file and the canonical circuit record before starting an experiment. Renaming a script does not "
        "make it new: the protocol key ignores claim revision and execution details, while the execution key binds "
        "the exact split, seed, checkpoint, and artifacts.", "",
        f"{report['event_count']} events; {report['protocol_count']} scientific protocols; "
        f"{report['execution_count']} exact executions; "
        f"{len(report['open_preregistrations'])} open preregistrations; "
        f"{len(report['duplicate_execution_groups'])} duplicate execution-key groups; "
        f"{len(report['protocol_repeat_review'])} protocol groups needing review.", "",
        "## Open preregistrations", "",
        "| circuit | event | test | claim | protocol |", "|---|---|---|---|---|",
    ]
    for row in report["open_preregistrations"]:
        lines.append(
            f"| `{cell(row['circuit'])}` | `{cell(row['event_id'])}` | {cell(row['test_type'])} | "
            f"`{cell(row['claim_id'])}` | `{row['protocol_key'][:12]}` |"
        )
    if not report["open_preregistrations"]:
        lines.append("| — | — | — | — | — |")

    lines.extend(["", "## Repeats requiring review", ""])
    if report["duplicate_execution_groups"]:
        lines.append("Exact execution-key duplicates: " + "; ".join(
            ", ".join(group) for group in report["duplicate_execution_groups"]
        ))
    else:
        lines.append("No two registered events have the same execution key.")
    if report["protocol_repeat_review"]:
        lines.extend(["", "| protocol | events | unlinked completed | planned duplicate |", "|---|---|---|---|"])
        for item in report["protocol_repeat_review"]:
            lines.append(
                f"| `{item['protocol_key'][:12]}` | {cell(', '.join(item['event_ids']))} | "
                f"{cell(', '.join(item['unlinked_completed_event_ids']))} | "
                f"{cell(', '.join(item['planned_duplicate_event_ids']))} |"
            )
    else:
        lines.extend(["", "No repeated scientific protocol currently lacks an explicit supersession or replication link."])

    lines.extend([
        "", "## All registered evidence events", "",
        "| circuit | event | stage/verdict | test | claim | protocol | design | execution | relation |", 
        "|---|---|---|---|---|---|---|---|---|",
    ])
    for row in events:
        relation = (
            f"supersedes `{row['supersedes_event_id']}`" if row["supersedes_event_id"] else
            f"replicates `{row['replicates_event_id']}`" if row["replicates_event_id"] else "—"
        )
        lines.append(
            f"| `{cell(row['circuit'])}` | `{cell(row['event_id'])}` | {row['stage']}/{row['verdict']} | "
            f"{row['test_type']} | `{row['claim_id']}` | `{row['protocol_key'][:12]}` | "
            f"`{row['design_key'][:12]}` | `{row['execution_key'][:12]}` | {relation} |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(
        f"wrote {OUT_MD.name} and {OUT_JSON.relative_to(ROOT)}: "
        f"{report['event_count']} events, {len(report['open_preregistrations'])} open, "
        f"{len(report['protocol_repeat_review'])} review groups"
    )


if __name__ == "__main__":
    main()
