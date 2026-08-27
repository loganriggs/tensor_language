#!/usr/bin/env python3
"""Authoritative no-network FineWeb oracle with exact same-state code handoff.

Preconditions: active streaming lanes are clear and
``prepare_fineweb_oracle_rows.py`` accepts a content-addressed identity gate
(real-stream tensor identity or pinned ordered-manifest/local-parquet identity).
This pipeline freezes the derived ship state before scoring, upgrades the
FineWeb null decision to the exact 20-null test, and conditionally invokes the
code-OOD callback without rebuilding or mutating the ship realization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
CANONICAL_FINEWEB = BQ / "ship_content_oracle_screen_results.json"
PRELIMINARY_ARCHIVE = BQ / "ship_content_oracle_screen_preliminary_results.json"
FROZEN_STATE = Path("/workspace/runs/bilin18_frozen_ship_v2.pt")
FROZEN_MANIFEST = Path("/workspace/runs/bilin18_frozen_ship_v2_manifest.json")
FROZEN_LOCK = Path("/workspace/runs/.bilin18_frozen_ship_v2.lock")
SHIP_SEED = 27182818
SENTINEL_ROWS = 2
SENTINEL_POSITIONS = (64, 127, 191, 255)

sys.path.insert(0, str(HERE))
import code_ood_oracle as code_oracle  # noqa: E402
import prepare_fineweb_oracle_rows as row_prep  # noqa: E402
import source_global_preflight  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(child) for child in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported frozen ship state: {type(value)}")


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def frozen_state_claim():
    """Serialize every creator or loader of the canonical frozen realization."""

    try:
        FROZEN_LOCK.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"canonical frozen-state operation already claimed: {FROZEN_LOCK}") from error
    try:
        yield
    finally:
        FROZEN_LOCK.rmdir()


def device_tree(value: Any, device: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device).contiguous().clone()
    if isinstance(value, dict):
        return {key: device_tree(child, device) for key, child in value.items()}
    if isinstance(value, list):
        return [device_tree(child, device) for child in value]
    if isinstance(value, tuple):
        return tuple(device_tree(child, device) for child in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported restored ship state: {type(value)}")


def _logical_receipt_sha256(row_receipt: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(row_receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_frozen_pair(row_receipt: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state_exists, manifest_exists = FROZEN_STATE.is_file(), FROZEN_MANIFEST.is_file()
    if state_exists != manifest_exists:
        raise RuntimeError(
            "inconsistent canonical frozen-state pair: "
            f"state_exists={state_exists} manifest_exists={manifest_exists}"
        )
    if not state_exists:
        raise RuntimeError("canonical frozen-state pair is absent")
    manifest = json.loads(FROZEN_MANIFEST.read_text())
    if manifest.get("schema_version") != 2:
        raise RuntimeError("canonical frozen-state manifest schema changed")
    if manifest.get("artifact_path") != str(FROZEN_STATE):
        raise RuntimeError("canonical frozen-state manifest path changed")
    if manifest.get("artifact_bytes") != FROZEN_STATE.stat().st_size:
        raise RuntimeError("canonical frozen-state artifact size changed")
    artifact_hash = file_sha256(FROZEN_STATE)
    if manifest.get("artifact_sha256") != artifact_hash:
        raise RuntimeError("canonical frozen-state artifact hash changed")
    if manifest.get("row_receipt_sha256") != _logical_receipt_sha256(row_receipt):
        raise RuntimeError("canonical frozen-state row receipt changed")
    payload = torch.load(FROZEN_STATE, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 2 or payload.get("ship_seed") != SHIP_SEED:
        raise RuntimeError("canonical frozen-state payload metadata changed")
    realization_hash = code_oracle.tensor_tree_sha256(payload.get("state"))
    if not (
        payload.get("ship_realization_sha256")
        == manifest.get("ship_realization_sha256")
        == realization_hash
    ):
        raise RuntimeError("canonical frozen-state realization tree changed")
    fingerprint = payload.get("baseline_fingerprint")
    if not isinstance(fingerprint, dict) or not isinstance(
        fingerprint.get("full_logits_raw_sha256"), str
    ):
        raise RuntimeError("canonical frozen-state baseline fingerprint is invalid")
    return payload, manifest


def validate_frozen_ship_pair(
    row_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with frozen_state_claim():
        return _load_frozen_pair(row_receipt)


def archive_preliminary_result() -> str | None:
    if not CANONICAL_FINEWEB.exists():
        return None
    current = json.loads(CANONICAL_FINEWEB.read_text())
    if current.get("config", {}).get("status") == "authoritative_frozen_ship_v2":
        raise RuntimeError("authoritative FineWeb v2 result already exists")
    current_hash = file_sha256(CANONICAL_FINEWEB)
    if PRELIMINARY_ARCHIVE.exists():
        if file_sha256(PRELIMINARY_ARCHIVE) != current_hash:
            raise RuntimeError("preliminary archive exists with different contents")
    else:
        shutil.copy2(CANONICAL_FINEWEB, PRELIMINARY_ARCHIVE)
    return current_hash


@torch.no_grad()
def baseline_fingerprint(sa: Any, twall: dict, all_attention: frozenset[int], rows: torch.Tensor) -> dict[str, Any]:
    sa.ORACLE_CORR.update({"on": False, "capture": None})
    sa.CONTENT_CORR["on"] = False
    batch = rows[:SENTINEL_ROWS].to(sa.DEV)
    idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
    logits = sa.fwd_arm(idx, all_attention, twall, frozenset(range(18))).float()
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).view_as(targets)
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[:, :64] = False
    logits_cpu = logits.detach().cpu().contiguous()
    sample = logits_cpu[:, list(SENTINEL_POSITIONS), :256].contiguous()
    return {
        "rows": SENTINEL_ROWS,
        "positions": list(SENTINEL_POSITIONS),
        "vocab_slice": [0, 256],
        "global_ce": float(ce[valid].mean()),
        "full_logits_raw_sha256": code_oracle.tensor_sha256(logits_cpu),
        "sample_logits": sample,
    }


def freeze_ship_realization(
    sa: Any, twall: dict, all_attention: frozenset[int],
    row_receipt: dict[str, Any], sentinel_rows: torch.Tensor,
) -> tuple[str, dict[str, Any]]:
    with frozen_state_claim():
        if FROZEN_STATE.exists() or FROZEN_MANIFEST.exists():
            raise RuntimeError("frozen ship artifact/manifest already exists")
        state_for_hash = {
            "TWALL": twall,
            "SHIP": sa.SHIP,
            "CORR": {key: sa.CORR[key] for key in ("on", "b", "U", "V")},
            "all_attention": sorted(all_attention),
        }
        realization_hash = code_oracle.tensor_tree_sha256(state_for_hash)
        fingerprint = baseline_fingerprint(sa, twall, all_attention, sentinel_rows)
        payload = {
            "schema_version": 2,
            "ship_realization_sha256": realization_hash,
            "ship_seed": SHIP_SEED,
            "state": cpu_tree(state_for_hash),
            "baseline_fingerprint": cpu_tree(fingerprint),
        }
        atomic_torch_save(payload, FROZEN_STATE)
        artifact_hash = file_sha256(FROZEN_STATE)
        manifest = {
            "schema_version": 2,
            "artifact_path": str(FROZEN_STATE),
            "artifact_sha256": artifact_hash,
            "artifact_bytes": FROZEN_STATE.stat().st_size,
            "ship_realization_sha256": realization_hash,
            "ship_seed": SHIP_SEED,
            "row_receipt_sha256": _logical_receipt_sha256(row_receipt),
            "baseline_fingerprint": {
                key: value for key, value in fingerprint.items() if key != "sample_logits"
            },
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=HERE, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "pipeline_sha256": file_sha256(Path(__file__)),
        }
        atomic_json_save(manifest, FROZEN_MANIFEST)
        _load_frozen_pair(row_receipt)
        return realization_hash, manifest


def restore_ship_realization(
    sa: Any, twall: dict, all_attention: frozenset[int],
    row_receipt: dict[str, Any], sentinel_rows: torch.Tensor,
) -> tuple[str, dict[str, Any]]:
    """Restore and behaviorally verify the canonical realization in this process."""

    with frozen_state_claim():
        payload, manifest = _load_frozen_pair(row_receipt)
        state = payload["state"]
        expected_attention = list(state["all_attention"])
        if expected_attention != sorted(all_attention):
            raise RuntimeError("fresh callback attention topology differs from frozen state")
        restored_twall = device_tree(state["TWALL"], sa.DEV)
        restored_ship = device_tree(state["SHIP"], sa.DEV)
        restored_corr = device_tree(state["CORR"], sa.DEV)
        twall.clear()
        twall.update(restored_twall)
        sa.SHIP.clear()
        sa.SHIP.update(restored_ship)
        sa.CORR.clear()
        sa.CORR.update(restored_corr)
        current_state = {
            "TWALL": twall,
            "SHIP": sa.SHIP,
            "CORR": {key: sa.CORR[key] for key in ("on", "b", "U", "V")},
            "all_attention": sorted(all_attention),
        }
        realization_hash = code_oracle.tensor_tree_sha256(current_state)
        if realization_hash != payload["ship_realization_sha256"]:
            raise RuntimeError("restored ship realization tree differs from frozen payload")
        observed = baseline_fingerprint(sa, twall, all_attention, sentinel_rows)
        expected = payload["baseline_fingerprint"]
        if observed["full_logits_raw_sha256"] != expected["full_logits_raw_sha256"]:
            raise RuntimeError("restored ship baseline logits differ from frozen fingerprint")
        if abs(float(observed["global_ce"]) - float(expected["global_ce"])) > 1e-12:
            raise RuntimeError("restored ship baseline CE differs from frozen fingerprint")
        if not torch.equal(observed["sample_logits"], expected["sample_logits"]):
            raise RuntimeError("restored ship sampled logits differ from frozen fingerprint")
        return realization_hash, manifest


def obtain_ship_realization(
    sa: Any, twall: dict, all_attention: frozenset[int],
    row_receipt: dict[str, Any], sentinel_rows: torch.Tensor,
) -> tuple[str, dict[str, Any], str]:
    state_exists, manifest_exists = FROZEN_STATE.exists(), FROZEN_MANIFEST.exists()
    if state_exists != manifest_exists:
        raise RuntimeError("inconsistent canonical frozen-state pair blocks obtain")
    if state_exists:
        realization_hash, manifest = restore_ship_realization(
            sa, twall, all_attention, row_receipt, sentinel_rows
        )
        return realization_hash, manifest, "restored"
    realization_hash, manifest = freeze_ship_realization(
        sa, twall, all_attention, row_receipt, sentinel_rows
    )
    return realization_hash, manifest, "created"


def exact_fineweb_decisions(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decisions = result["site_decisions"]
    for site in (0, 1, 2):
        key = str(site)
        content_gain = result["paired_gains"][key]["heldout"]["content"]["global"]["mean"]
        null_gains = [
            result["paired_gains"][key]["heldout"][f"null_{index:02d}"]["global"]["mean"]
            for index in range(20)
        ]
        exact = code_oracle.exact_null_test(content_gain, null_gains)
        decisions[key]["preliminary_interpolated_null95_gate"] = decisions[key][
            "content_beats_matched_null95_heldout"
        ]
        decisions[key]["exact_twenty_null_test"] = exact
        decisions[key]["content_beats_matched_null95_heldout"] = exact["passes_5pct"]
    result["training_license_sites"] = [
        site for site in (0, 1, 2)
        if decisions[str(site)]["full_oracle_ci95_lower_gt_zero"]
        and decisions[str(site)]["content_positive_both_splits"]
        and decisions[str(site)]["exact_twenty_null_test"]["passes_5pct"]
    ]
    return decisions


def main() -> None:
    source_global_preflight.require_defined_globals([
        BQ / "ship_error_attrib.py",
        HERE / "code_ood_oracle.py",
        HERE / "prepare_fineweb_oracle_rows.py",
        Path(__file__),
    ])
    archive_hash = archive_preliminary_result()
    row_receipt, frozen_rows = row_prep.validate_receipt()
    code_rows, code_manifest = code_oracle.load_frozen_corpus()

    def frozen_fineweb_rows(n: int = 120, skip: int = 0) -> torch.Tensor:
        spec = (n, skip)
        if spec not in frozen_rows:
            raise RuntimeError(f"unregistered FineWeb row request in oracle v2: {spec}")
        return frozen_rows[spec].clone()

    torch.manual_seed(SHIP_SEED)
    torch.cuda.manual_seed_all(SHIP_SEED)
    sys.path.insert(0, str(BQ))
    import ship_error_attrib as sa  # noqa: PLC0415

    torch.manual_seed(SHIP_SEED)
    torch.cuda.manual_seed_all(SHIP_SEED)
    sa.cl.fineweb_rows = frozen_fineweb_rows
    original_callback = sa.run_oracle_content_screen

    def authoritative_callback(
        twall: dict, all_attention: frozenset[int], start_time: float
    ) -> None:
        realization_hash, frozen_manifest, lifecycle = obtain_ship_realization(
            sa, twall, all_attention, row_receipt, code_rows
        )
        original_callback(twall, all_attention, start_time)
        result = json.loads(CANONICAL_FINEWEB.read_text())
        exact_fineweb_decisions(result)
        result["config"].update({
            "status": "authoritative_frozen_ship_v2",
            "authority": "canonical_fineweb",
            "authorized_for_scored_experiments": True,
            "ship_realization_sha256": realization_hash,
            "ship_seed": SHIP_SEED,
            "frozen_ship_artifact_sha256": frozen_manifest["artifact_sha256"],
            "frozen_ship_lifecycle": lifecycle,
            "fineweb_row_receipt_sha256": frozen_manifest["row_receipt_sha256"],
            "preliminary_result_sha256": archive_hash,
            "null_gate": "exact one-sided Monte Carlo; content beats all 20",
        })
        temporary = CANONICAL_FINEWEB.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, CANONICAL_FINEWEB)
        sites = result["training_license_sites"]
        print(json.dumps({
            "authoritative_training_license_sites": sites,
            "ship_realization_sha256": realization_hash,
        }, indent=2), flush=True)
        if sites:
            code_oracle.run_code_oracle(
                sa, twall, all_attention, start_time, sites, result,
                code_rows, code_manifest,
            )
        else:
            print("FineWeb v2 licensed no site; code OOD stage correctly skipped", flush=True)

    sa.run_oracle_content_screen = authoritative_callback
    sa.main(oracle_content_screen=True)


if __name__ == "__main__":
    main()
