import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import tiktoken
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "freeze_code_oracle_corpus_v2.py"
SPEC = importlib.util.spec_from_file_location("freeze_code_oracle_corpus_v2", SCRIPT)
FREEZE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(FREEZE)


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, check=True,
        stdout=subprocess.PIPE,
    ).stdout


def test_v2_corpus_is_reconstructible_file_disjoint_and_file_contained():
    payload = torch.load(HERE / "code_oracle_corpus_v2.pt", weights_only=False)
    manifest = json.loads((HERE / "code_oracle_corpus_v2_manifest.json").read_text())
    rows = payload["rows"]
    assert payload["manifest"] == manifest
    assert manifest["schema_version"] == 2
    assert tuple(rows.shape) == (480, 257)
    assert rows.dtype == torch.long
    assert manifest["source_commit"] == FREEZE.SOURCE_COMMIT
    assert manifest["splits"] == {
        "basis": [0, 96], "discovery": [96, 288], "heldout": [288, 480]
    }
    assert manifest["no_row_crosses_file_boundary"] is True
    assert manifest["file_disjoint_splits"] is True
    assert min(manifest["split_cluster_counts"].values()) >= 24
    assert manifest["tensor_raw_sha256"] == hashlib.sha256(
        rows.numpy().tobytes(order="C")
    ).hexdigest()
    assert manifest["construction_script_sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()
    assert int(rows.min()) >= 0 and int(rows.max()) < 50257
    assert len({tuple(row.tolist()) for row in rows}) == len(rows)

    split_paths = {}
    encoder = tiktoken.get_encoding("gpt2")
    blob_cache = {}
    row_index = 0
    for split in ("basis", "discovery", "heldout"):
        provenance = manifest["row_provenance"][split]
        start, end = manifest["splits"][split]
        assert len(provenance) == end - start
        split_paths[split] = {row["path"] for row in provenance}
        for row_meta in provenance:
            path = row_meta["path"]
            if path not in blob_cache:
                blob = git_blob(manifest["source_commit"], path)
                assert hashlib.sha256(blob).hexdigest() == row_meta["blob_sha256"]
                blob_cache[path] = [encoder.eot_token] + encoder.encode_ordinary(
                    blob.decode("utf-8", errors="replace")
                )
            tokens = blob_cache[path]
            lo, hi = row_meta["token_start"], row_meta["token_end"]
            assert hi - lo == 257
            assert hi <= len(tokens)
            assert torch.equal(rows[row_index], torch.tensor(tokens[lo:hi]))
            row_index += 1
    assert row_index == len(rows)
    assert split_paths["basis"].isdisjoint(split_paths["discovery"])
    assert split_paths["basis"].isdisjoint(split_paths["heldout"])
    assert split_paths["discovery"].isdisjoint(split_paths["heldout"])


def test_v2_paths_obey_frozen_hash_assignment_and_file_cap():
    manifest = json.loads((HERE / "code_oracle_corpus_v2_manifest.json").read_text())
    for split, records in manifest["files"].items():
        for record in records:
            assigned, bucket = FREEZE.split_for_path(record["path"])
            assert assigned == split
            assert bucket == record["assignment_bucket"]
            assert 1 <= record["rows_used"] <= FREEZE.ROWS_PER_FILE_CAP
