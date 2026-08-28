import ast
import json

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
