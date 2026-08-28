import ast
import hashlib
import json

import torch

from . import affine_codec
from . import mlp4_bilinear_residual_codec as native_codec
from . import mlp4_seeded_random_bilinear_codec as random_codec
from . import mlp4_z4_validation_protocol as protocol


def test_paired_row_ci_has_expected_center_and_width_without_loading_model():
    # Extract only the pure helper to avoid importing the model-bearing runner.
    source = (protocol.HERE/"mlp4_z4_validation.py").read_text()
    tree = ast.parse(source)
    node = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "paired_row_ci95")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, "<paired_row_ci95>", "exec"), namespace)
    result = namespace["paired_row_ci95"]([1.0, 2.0, 3.0, 4.0])
    assert result["mean"] == 2.5 and result["clusters"] == 4
    assert result["low"] < result["mean"] < result["high"]
    assert abs((result["high"]-result["mean"])
               - (result["mean"]-result["low"])) < 1e-12


def test_protocol_and_runner_boundary():
    p = protocol.load_and_validate()
    ast.parse((protocol.HERE/"mlp4_z4_validation.py").read_text())
    inventory = json.loads((protocol.HERE/"mlp4_z4_candidate_inventory.json").read_text())
    assert p["candidate_order"] == [row["candidate_id"] for row in inventory["candidates"]]
    assert len(inventory["native_random_actual_bit_pairings"]) == 5


def test_each_runtime_program_bypasses_native_mlp4():
    source = (protocol.HERE/"mlp4_z4_validation.py").read_text()
    tree = ast.parse(source)
    forward = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "forward_inline")
    text = ast.unparse(forward)
    assert "execute(program, z)" in text
    assert text.index("execute(program, z)") < text.index("block.mlp(z)")


def test_clean_runtime_has_no_transitive_experiment_or_row_imports():
    source = (protocol.HERE/"bilin18_clean_runtime.py").read_text()
    tree = ast.parse(source)
    imported = {alias.name for node in tree.body if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert imported == {"__future__", "hashlib", "json", "sys", "pathlib",
                        "torch", "huggingface_hub", "jacclust.tt_model"}
    assert "local_files_only=True" in source
    tree = ast.parse(source)
    top_level_text = "\n".join(ast.unparse(node) for node in tree.body
                               if not isinstance(node, (ast.FunctionDef, ast.ClassDef)))
    for forbidden_call in ("local_blob(", "TT.GPT(", "torch.load(", "initialize("):
        assert forbidden_call not in top_level_text


def test_every_frozen_stream_decodes_to_the_promised_z4_interface():
    artifact = torch.load(protocol.HERE/"mlp4_z4_candidate_bytes.pt",
                          map_location="cpu", weights_only=False)
    inventory = json.loads(
        (protocol.HERE/"mlp4_z4_candidate_inventory.json").read_text())
    assert set(artifact["encoded"]) == {
        row["candidate_id"] for row in inventory["candidates"]}
    for row in inventory["candidates"]:
        candidate_id = row["candidate_id"]
        encoded = artifact["encoded"][candidate_id]
        assert "sha256:"+hashlib.sha256(encoded).hexdigest() == \
            row["canonical_bytes_hash"]
        if row["family"] == "linear":
            decoded = affine_codec.decode_affine(encoded)
            assert decoded["weight"].shape == (1152, 1152)
            assert decoded["bias"].shape == (1152,)
        elif row["family"] == "native_product":
            decoded = native_codec.decode(encoded)
            assert decoded["A"].shape == (1152, row["capacity"])
            assert decoded["B"].shape == decoded["A"].shape
            assert decoded["C"].shape == (row["capacity"], 1152)
            assert decoded["bias"].shape == (1152,)
        else:
            decoded = random_codec.decode(encoded)
            assert decoded["din"] == 1152
            assert decoded["components"] == row["capacity"]
            assert decoded["C"].shape == (row["capacity"], 1152)
            assert decoded["bias"].shape == (1152,)


def test_resume_validator_is_pure_and_rejects_nonprefix_or_completed_state(tmp_path):
    source = (protocol.HERE/"mlp4_z4_validation.py").read_text()
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"sha", "atomic_json", "validate_resume"}]
    namespace = {"PROTOCOL": tmp_path/"protocol.json", "hashlib": hashlib,
                 "json": json}
    namespace["PROTOCOL"].write_text("{}")
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<resume>", "exec"), namespace)
    p = {"protocol_id": "p", "candidate_order": ["a", "b"],
         "pinned_artifacts": {"x": "y"}}
    inventory = {key: {"canonical_bytes_hash": "sha256:"+key} for key in ("a", "b")}
    base = {"partial": True, "protocol_id": "p",
            "protocol_sha256": namespace["sha"](namespace["PROTOCOL"]),
            "pinned_artifacts": {"x": "y"}, "live_rows": [0.]*960,
            "anchor_rows": [0.]*960, "points": [], "row_scores_by_id": {}}
    assert namespace["validate_resume"](base, p, inventory) == ([], {})
    good_point = {"candidate_id": "a", "program_hash": "sha256:a"}
    good = {**base, "points": [good_point],
            "row_scores_by_id": {"a": [0.]*960}}
    assert namespace["validate_resume"](good, p, inventory) == (
        [good_point], {"a": [0.]*960})
    ledger = tmp_path/"ledger.json"
    namespace["atomic_json"](ledger, good)
    assert json.loads(ledger.read_text()) == good
    assert not (tmp_path/"ledger.json.tmp").exists()
    completed = {**base, "partial": False}
    try:
        namespace["validate_resume"](completed, p, inventory)
    except RuntimeError:
        pass
    else:
        raise AssertionError("completed result accepted for rerun")
    bad = {**base, "points": [
        {"candidate_id": "b", "program_hash": "sha256:b"}],
        "row_scores_by_id": {"b": [0.]*960}}
    try:
        namespace["validate_resume"](bad, p, inventory)
    except ValueError as error:
        assert "prefix" in str(error)
    else:
        raise AssertionError("nonprefix partial state accepted")
