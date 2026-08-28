"""Import-free regression tests for table_vs_site's expensive arm dispatcher."""

import ast
import inspect
from pathlib import Path

import pytest
import torch


SOURCE = Path(__file__).with_name("table_vs_site.py")


def _main_and_run_g():
    tree = ast.parse(SOURCE.read_text())
    main = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    run_g = next(
        node for node in main.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_g"
    )
    return main, run_g


def _isolated_function(function_node, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[function_node], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace[function_node.name]


def _top_level_function(name):
    tree = ast.parse(SOURCE.read_text())
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _empirical_fixture():
    probes = [("mlp", 5), ("mlp", 4)]

    class Handle:
        def __init__(self, module):
            self.module = module

        def remove(self):
            self.module.hook = None

    class Module:
        hook = None

        def register_forward_hook(self, hook):
            assert self.hook is None
            self.hook = hook
            return Handle(self)

    modules = {site: Module() for site in probes}
    state = {}

    def mod_of(*site):
        return modules[site]

    def forward_logits(idx):
        state["idx"] = idx
        for offset, site in enumerate(probes, start=1):
            output = idx.unsqueeze(-1).expand(-1, -1, 2).double() + offset
            modules[site].hook(None, (), output)
        return torch.empty(0)

    namespace = {
        "torch": torch,
        "V": 6,
        "D": 2,
        "DEV": torch.device("cpu"),
        "NCOV": 3,
        "STATE": state,
        "mod_of": mod_of,
        "forward_logits": forward_logits,
    }
    empirical_rows = _isolated_function(_top_level_function("empirical_rows"), namespace)
    base_bank = {
        site: -torch.arange(1, 13, dtype=torch.float32).reshape(6, 2) - offset
        for offset, site in enumerate(probes)
    }
    covered = torch.tensor([False, True, True, True, False, False])
    rows = torch.zeros((2, 67), dtype=torch.long)
    rows[0, 64:66] = torch.tensor([1, 2])
    rows[1, 64:66] = torch.tensor([3, 4])  # token 4 is observed but outside frozen coverage
    return empirical_rows, rows, probes, base_bank, covered


def test_every_run_g_arm_binds_and_empirical_override_is_reachable():
    main, run_g_node = _main_and_run_g()
    parameter_names = [arg.arg for arg in run_g_node.args.args]
    assert parameter_names == ["label", "hooked", "gains", "override"]
    assert len(run_g_node.args.defaults) == 1
    assert isinstance(run_g_node.args.defaults[0], ast.Constant)
    assert run_g_node.args.defaults[0].value is None

    signature = inspect.Signature([
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          default=None if name == "override" else inspect.Parameter.empty)
        for name in parameter_names
    ])
    calls = [
        node for node in ast.walk(main)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "run_g"
    ]
    assert calls
    for call in calls:
        signature.bind(*([object()] * len(call.args)),
                       **{keyword.arg: object() for keyword in call.keywords})

    target = ("mlp", 5)
    seen = []

    def evaluate(rows, hooks, keep_mask):
        seen.append((rows, hooks, keep_mask))
        return {"top1": 0.0}

    namespace = {
        "allh": {target: "length-1-row-hook"},
        "H": [],
        "gain_hook": lambda gain: gain,
        "evs": {"skip7000": "rows"},
        "evaluate": evaluate,
        "keep_mask": "mask",
        "res": {},
    }
    run_g = _isolated_function(run_g_node, namespace)
    run_g("EM_mlp5_raw", [target], {}, {target: "empirical-row-hook"})

    assert seen == [("rows", [(target, "empirical-row-hook")], "mask")]
    assert namespace["res"]["skip7000"]["EM_mlp5_raw"] == {"top1": 0.0}


def test_empirical_rows_changes_exact_frozen_support_and_preserves_fallback_bytes():
    empirical_rows, rows, probes, base_bank, covered = _empirical_fixture()
    output, changed = empirical_rows(rows, probes, base_bank, covered)

    for site in probes:
        actual_changed = (output[site] != base_bank[site]).any(dim=1)
        assert torch.equal(actual_changed, covered)
        assert torch.equal(output[site][~covered], base_bank[site][~covered])
        assert changed[site] == 3


def test_empirical_rows_fails_before_building_bank_when_covered_token_is_unobserved():
    empirical_rows, rows, probes, base_bank, covered = _empirical_fixture()
    rows[1, 64] = 2  # remove the only occurrence of frozen-covered token 3

    with pytest.raises(AssertionError, match="observed 2 of 3 frozen covered tokens"):
        empirical_rows(rows, probes, base_bank, covered)
