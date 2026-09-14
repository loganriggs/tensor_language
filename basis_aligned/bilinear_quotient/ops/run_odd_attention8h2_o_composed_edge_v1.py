#!/usr/bin/env python3
# BQGATE:192bodyforwards;96prefixes;180seconds;no fitting.
"""pred_a parent score replay; pred_b live/exact execution; pred_c behavior/control replay; pred_d unfavorable price."""
import json, os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F

from even_value_shared_graph_v1 import SharedGraph
from odd_attention8h2_o_composed_edge_v1 import composed_o_delta
from odd_contextual_positions_v1 import contextual_masks
from odd_framing_role_split_v1 import role_masks
from regional_cue_row_check_v1 import validate
from run_even_value_factorial_native_v1 import cpu_control
from sparse_path_stability_atlas_v1 import digest

STEM = "ODD_ATTENTION8H2_O_COMPOSED_EDGE_V1"
READOUTS = [
    ("target", None), ("work_jobs", (670, 3946)), ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
]


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(path) == value for path, value in binding.items())
    rows = json.loads((P / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ROWS.json").read_text())["rows"]
    assert len(rows) == 96
    validate(rows)
    contexts, roles = contextual_masks(rows), role_masks(rows)
    prior = torch.load(P / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)["values"]
    edge_cpu = torch.load(P / "ODD_ATTENTION8H2_ROUTING_CLOSURE_V1_PROGRAM.pt", map_location="cpu", weights_only=True)
    graph_cpu = torch.load(P / "EVEN_VALUE_SHARED_GRAPH_PACKED_V1_PROGRAM.pt", map_location="cpu", weights_only=True)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        assert json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())["pred_a"]
        print("192bodyforwards;native/composedO;CPUcontrol", cpu_control())
        return
    out, program_path, artifact_path = P / (STEM + "_RESULT.json"), P / (STEM + "_PROGRAM.pt"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not program_path.exists() and not artifact_path.exists()
    start = time.perf_counter()
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    edge = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in edge_cpu.items()}
    graph = SharedGraph({key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in graph_cpu.items()})
    context = {"arm": 0}
    a8inputs, deltas = [], []
    representatives = []

    def a8pre(module, args):
        if context["arm"] == 0:
            a8inputs.append(args[0].detach().cpu())

    def b9pre(module, args):
        if context["arm"] == 0:
            return None
        i, donor = context["i"], context["i"] ^ 1
        x, first, x0 = args
        city = int(torch.nonzero(contexts[i]["city"])[0])
        ids0 = torch.tensor([rows[i]["ids"]], device="cuda")
        idsd = torch.tensor([rows[donor]["ids"]], device="cuda")
        recipient_embedding = F.rms_norm(model.transformer.wte(ids0), (1152,))[:, city]
        donor_embedding = F.rms_norm(model.transformer.wte(idsd), (1152,))[:, city]
        delta = composed_o_delta(
            edge, graph, a8inputs[i].to("cuda"), city, recipient_embedding,
            donor_embedding, x, x0, first, module.lambdas,
            roles[i]["framing"].to("cuda"),
        )
        context["delta"] = delta
        deltas.append(float(delta.norm()))
        if rows[i]["endpoint"] == 0 and rows[i]["pair"] == 0 and rows[i]["cue"] == "British":
            representatives.append({
                "family": rows[i]["family"], "block8_current": a8inputs[i], "city_index": city,
                "recipient_embedding": recipient_embedding.cpu(), "donor_embedding": donor_embedding.cpu(),
                "block9_residual": x.detach().cpu(), "initial": x0.detach().cpu(),
                "first_values": first.detach().cpu(), "reentry": module.lambdas.detach().cpu(),
                "framing_mask": roles[i]["framing"], "delta": delta.cpu(),
            })
        return None

    def a9out(module, args, output):
        if context["arm"] == 0:
            return output
        return output[0] + context["delta"].to(output[0].dtype), output[1]

    hooks = [
        model.transformer.h[8].attn.register_forward_pre_hook(a8pre),
        model.transformer.h[9].register_forward_pre_hook(b9pre),
        model.transformer.h[9].attn.register_forward_hook(a9out),
    ]
    values = torch.zeros(2, 96, 6, dtype=torch.float64)
    count = 0

    def forward(i, arm):
        nonlocal count
        row = rows[i]
        ids = torch.tensor([row["ids"]], device="cuda")
        x = F.rms_norm(model.transformer.wte(ids), (1152,))
        x0, first = x, None
        for block in model.transformer.h:
            x, first = block(x, first, x0)
        scores = (30 * torch.tanh(model.lm_head(F.rms_norm(x[:, -1], (1152,))) / 30))[0]
        pairs = [(row["uk_id"], row["us_id"])] + [pair for _, pair in READOUTS[1:]]
        for j, (left, right) in enumerate(pairs):
            values[arm, i, j] = (scores[left] - scores[right]).cpu()
        count += 1

    try:
        for i in range(96):
            forward(i, 0)
        assert len(a8inputs) == 96
        context["arm"] = 1
        for i in range(96):
            context["i"] = i
            forward(i, 1)
    finally:
        for hook in hooks:
            hook.remove()
    assert count == 192 and bool(torch.isfinite(values).all())
    anchor = max(float((values[0, :, :2] - prior[0, :, :2]).abs().max()), float((values[1, :, :2] - prior[1, :, :2]).abs().max()))
    effect = values[1] - values[0]
    cells = []
    for family in range(4):
        ix = [i for i, row in enumerate(rows) if row["family"] == family]
        native = values[0, ix[::2], 0] - values[0, ix[1::2], 0]
        cue = effect[ix[::2], 0] - effect[ix[1::2], 0]
        target = float(cue.square().mean().sqrt())
        prior_effect = prior[1, ix] - prior[0, ix]
        control_errors = {READOUTS[j][0]: float((effect[ix, j] - prior_effect[:, j]).abs().max()) for j in range(2, 6)}
        controls = {READOUTS[j][0]: float(effect[ix, j].square().mean().sqrt() / max(target, 1e-8)) for j in range(2, 6)}
        cells.append({
            "family": family, "native_positive_pairs": int((native > 0).sum()),
            "composed_target_rms": target, "control_ratios_to_target": controls,
            "max_parent_control_replay_error": max(control_errors.values()),
            "capability_pass": int((native > 0).sum()) >= 10 and target >= 1e-5,
            "control_replay_pass": max(control_errors.values()) <= 1e-5,
        })
    lengths = sorted(set(len(row["ids"]) for row in rows))
    prices = []
    for tokens in lengths:
        dedicated = 4 * tokens * 1152 + 2 * 1152 + tokens + 1
        dense = tokens * 1152
        prices.append({"tokens": tokens, "dedicated_runtime_input_scalars": dedicated, "dense_delta_scalars": dense, "dedicated_to_dense_ratio": dedicated / dense})
    program = {"edge": edge_cpu, "o_graph": graph_cpu, "reentry_note": "runtime two-scalar port"}
    torch.save(program, program_path)
    torch.save({"representatives": representatives}, artifact_path)
    result = {
        "pred_a": anchor <= 1e-5,
        "pred_b": len(deltas) == 96 and min(deltas) >= 1e-8 and all(torch.isfinite(torch.tensor(deltas))) and count == 192,
        "pred_c": all(cell["capability_pass"] and cell["control_replay_pass"] for cell in cells),
        "pred_d": all(item["dedicated_to_dense_ratio"] > 1 for item in prices),
        "max_parent_score_anchor_error": anchor, "min_composed_o_delta_norm": min(deltas),
        "families": cells, "prices": prices,
        "static_weight_scalars": sum(value.numel() for value in edge_cpu.values() if isinstance(value, torch.Tensor)) + sum(value.numel() for value in graph_cpu.values() if isinstance(value, torch.Tensor)) + 2,
        "body_forwards": count, "seconds": time.perf_counter() - start,
        "program_sha256": digest(program_path), "artifact_sha256": digest(artifact_path),
        "source_shas": binding,
        "scope": "Conditional composed extraction from block8/current city ports through head8.2 inherited write and head9.8-O current-value delta, with native suffix replay. Native state generation and suffix remain external.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
