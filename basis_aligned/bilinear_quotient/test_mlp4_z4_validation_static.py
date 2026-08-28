import ast
import hashlib
import json

import torch

from . import affine_codec
from . import mlp4_bilinear_residual_codec as native_codec
from . import mlp4_seeded_random_bilinear_codec as random_codec
from . import mlp4_z4_validation_protocol as protocol


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
