"""Import-free regression tests for table_vs_site's expensive arm dispatcher."""

import ast
import inspect
from pathlib import Path


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


def _isolated_run_g(run_g_node, namespace):
    module = ast.fix_missing_locations(ast.Module(body=[run_g_node], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["run_g"]


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
    run_g = _isolated_run_g(run_g_node, namespace)
    run_g("EM_mlp5_raw", [target], {}, {target: "empirical-row-hook"})

    assert seen == [("rows", [(target, "empirical-row-hook")], "mask")]
    assert namespace["res"]["skip7000"]["EM_mlp5_raw"] == {"top1": 0.0}
