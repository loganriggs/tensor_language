"""CPU tests for the physical Task 14 DISCOVERY endpoint boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import build_task14_projector_discovery_endpoint_shard as shard


def test_real_sources_emit_exact_discovery_endpoint_census_without_text() -> None:
    payload = shard.build_shard()
    assert payload["partition"] == "DISCOVERY"
    assert payload["endpoint_count"] == 128
    assert payload["group_count"] == 16
    endpoints = payload["endpoints"]
    assert len({row["endpoint_id"] for row in endpoints}) == 128
    assert len({row["group_id"] for row in endpoints}) == 16
    assert {row["partition"] for row in endpoints} == {"DISCOVERY"}
    assert all(row["final_position"] == len(row["ids"]) - 1 for row in endpoints)
    assert all("text" not in key for row in endpoints for key in row)


def test_validation_or_unknown_endpoint_fails_closed() -> None:
    payload = shard.build_shard()
    partition = json.loads(shard.SOURCE_PATHS["partition"].read_text())
    discovery = {
        row["group_id"] for row in partition["records"]
        if row["partition"] == "DISCOVERY"
    }
    validation = {
        row["group_id"] for row in partition["records"]
        if row["partition"] == "VALIDATION"
    }
    changed = copy.deepcopy(payload)
    changed["endpoints"][0]["group_id"] = sorted(validation)[0]
    with pytest.raises(shard.DiscoveryShardError, match="VALIDATION or unknown"):
        shard.validate_shard(
            changed, discovery_groups=discovery, validation_groups=validation
        )


def test_source_hash_mismatch_fails_before_parsing(tmp_path: Path) -> None:
    paths = dict(shard.SOURCE_PATHS)
    changed = tmp_path / "authority.json"
    changed.write_text("{}\n", encoding="utf-8")
    paths["authority"] = changed
    with pytest.raises(shard.DiscoveryShardError, match="immutable authority hash"):
        shard.build_shard(source_paths=paths)


def test_dry_run_is_summary_only_and_writes_nothing(capsys) -> None:
    assert shard.main(["--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["endpoint_count"] == 128
    assert "endpoints" not in output
    assert output["validation_endpoints_emitted"] == 0
    assert output["output_written"] is False
    assert output["model_loaded"] is False
    assert output["gpu_accessed"] is False
    assert output["queue_touched"] is False


def test_output_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "discovery.json"
    assert shard.main(["--output", str(output)]) == 0
    written = json.loads(output.read_text())
    assert written["endpoint_count"] == 128
    with pytest.raises(FileExistsError):
        shard.main(["--output", str(output)])


@pytest.mark.parametrize("argv", [[], ["--unknown"], ["--dry-run", "--output", "x"]])
def test_cli_requires_one_known_mode(argv) -> None:
    with pytest.raises(SystemExit):
        shard.main(argv)
