import hashlib
import json

import torch

from . import mlp_composition_capture as capture


def write_valid(tmp_path):
    contract = capture.load_contract()
    digest = "a"*64
    artifact = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "object_id": "blocks.3.mlp",
        "interface": "blocks.3.mlp.rmsnorm_input",
        "checkpoint_sha256": digest,
        "rows_artifact_sha256": "b"*64,
        "upstream_program_sha256": "c"*64,
        "observation_ids": torch.tensor([[0, 2], [0, 3], [1, 0]]),
        "z_live": torch.randn(3, 5),
        "z_composed": torch.randn(3, 5),
    }
    artifact_path = tmp_path/"capture.pt"
    torch.save(artifact, artifact_path)
    manifest = {
        "schema_version": 1, "model_id": "synthetic", "layer": 3,
        "checkpoint_revision": "pinned", "checkpoint_sha256": digest,
        "object_id": artifact["object_id"], "interface": artifact["interface"],
        "state_width": 5, "rows_artifact_sha256": "b"*64,
        "row_member": "synthetic_rows", "upstream_program_id": "prefix-v1",
        "upstream_program_sha256": "c"*64, "capture_source_sha256": "d"*64,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, artifact_path, manifest, artifact


def test_valid_capture_round_trip(tmp_path):
    manifest_path, artifact_path, manifest, artifact = write_valid(tmp_path)
    got_manifest, got_artifact = capture.load_and_validate(manifest_path, artifact_path)
    assert got_manifest == manifest
    assert torch.equal(got_artifact["observation_ids"], artifact["observation_ids"])


def test_rejects_hash_metadata_shape_order_and_nonfinite(tmp_path):
    manifest_path, artifact_path, manifest, artifact = write_valid(tmp_path)
    cases = []
    bad = dict(manifest); bad["artifact_sha256"] = "0"*64
    cases.append((bad, artifact))
    bad_artifact = dict(artifact); bad_artifact["interface"] = "blocks.4.mlp.rmsnorm_input"
    cases.append((manifest, bad_artifact))
    bad_artifact = dict(artifact); bad_artifact["z_composed"] = torch.randn(2, 5)
    cases.append((manifest, bad_artifact))
    bad_artifact = dict(artifact); bad_artifact["observation_ids"] = torch.tensor(
        [[0, 3], [0, 2], [1, 0]])
    cases.append((manifest, bad_artifact))
    bad_artifact = dict(artifact); bad_artifact["z_live"] = artifact["z_live"].clone()
    bad_artifact["z_live"][0, 0] = float("nan")
    cases.append((manifest, bad_artifact))
    for index, (bad_manifest, bad_artifact) in enumerate(cases):
        bad_path = tmp_path/f"bad-{index}.pt"; torch.save(bad_artifact, bad_path)
        bad_manifest = dict(bad_manifest)
        if index:
            bad_manifest["artifact_sha256"] = hashlib.sha256(bad_path.read_bytes()).hexdigest()
        bad_manifest_path = tmp_path/f"bad-{index}.json"
        bad_manifest_path.write_text(json.dumps(bad_manifest))
        try:
            capture.load_and_validate(bad_manifest_path, bad_path)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid capture case {index} was accepted")
