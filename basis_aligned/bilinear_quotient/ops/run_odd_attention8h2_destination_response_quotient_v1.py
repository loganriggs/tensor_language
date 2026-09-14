#!/usr/bin/env python3
# BQGATE:256bodyforwards;16prefixes;180seconds;no fitting outside frozen discovery split.
"""pred_a instrument; pred_b discovery operational quotient; pred_c heldout identification.
pred_d destination-effect composition; pred_e target/control selectivity.
"""
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F

from even_value_shared_graph_v1 import SharedGraph
from odd_attention8h2_o_composed_edge_v1 import composed_o_delta
from run_even_value_factorial_native_v1 import cpu_control
from sparse_path_stability_atlas_v1 import digest

STEM = "ODD_ATTENTION8H2_DESTINATION_RESPONSE_QUOTIENT_V1"
CONTROLS = [(3797, 3290), (2266, 4171), (3321, 3431), (17180, 10912)]


def rel(x, y):
    return float((x - y).norm() / y.norm().clamp_min(1e-8))


def cos(x, y):
    return float((x.flatten() @ y.flatten()) / (x.norm() * y.norm()).clamp_min(1e-8))


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(path) == value for path, value in binding.items())
    contexts = json.loads((P / (STEM + "_ROWS.json")).read_text())["contexts"]
    parent_rows = json.loads((P / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ROWS.json").read_text())["rows"]
    parent = torch.load(P / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)["values"]
    combined = torch.load(P / "ODD_ATTENTION8H2_O_COMPOSED_EDGE_V1_PROGRAM.pt", map_location="cpu", weights_only=True)
    assert len(contexts) == 16 and parent.shape == (4, 96, 6)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        control = json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())
        assert control["pred_a"] and control["contexts"] == 16
        print("256bodyforwards;16native+16full+224single;CPUcontrol", cpu_control())
        return
    out, artifact = P / (STEM + "_RESULT.json"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not artifact.exists()
    start = time.perf_counter()
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    edge = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in combined["edge"].items()}
    graph = SharedGraph({k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in combined["o_graph"].items()})
    context = {"arm": "native"}
    a8inputs, full_deltas, single_deltas = [], [], [[] for _ in contexts]

    def a8pre(module, args):
        if context["arm"] == "native":
            a8inputs.append(args[0].detach().cpu())

    def b9pre(module, args):
        if context["arm"] == "native":
            return None
        i, donor = context["i"], context["i"] ^ 1
        row = contexts[i]
        x, first, x0 = args
        city = row["city_position"]
        ids0 = torch.tensor([row["ids"]], device="cuda")
        idsd = torch.tensor([contexts[donor]["ids"]], device="cuda")
        recipient = F.rms_norm(model.transformer.wte(ids0), (1152,))[:, city]
        donor_embedding = F.rms_norm(model.transformer.wte(idsd), (1152,))[:, city]
        destination = torch.zeros(len(row["ids"]), dtype=torch.bool, device="cuda")
        if context["arm"] == "full":
            destination[row["destination_positions"]] = True
        else:
            destination[row["destination_positions"][context["destination"]]] = True
        delta = composed_o_delta(
            edge, graph, a8inputs[i].to("cuda"), city, recipient, donor_embedding,
            x, x0, first, module.lambdas, destination,
        )
        context["delta"] = delta
        if context["arm"] == "full":
            full_deltas.append(delta.detach().cpu())
        else:
            single_deltas[i].append(delta.detach().cpu())
        return None

    def a9out(module, args, output):
        if context["arm"] == "native":
            return output
        return output[0] + context["delta"].to(output[0].dtype), output[1]

    hooks = [
        model.transformer.h[8].attn.register_forward_pre_hook(a8pre),
        model.transformer.h[9].register_forward_pre_hook(b9pre),
        model.transformer.h[9].attn.register_forward_hook(a9out),
    ]
    native = torch.zeros(16, 10, dtype=torch.float64)
    full = torch.zeros_like(native)
    singles = [torch.zeros(len(row["destination_positions"]), 10, dtype=torch.float64) for row in contexts]
    count = 0

    def forward(i, target):
        nonlocal count
        row = contexts[i]
        ids = torch.tensor([row["ids"]], device="cuda")
        x = F.rms_norm(model.transformer.wte(ids), (1152,))
        x0, first = x, None
        for block in model.transformer.h:
            x, first = block(x, first, x0)
        scores = (30 * torch.tanh(model.lm_head(F.rms_norm(x[:, -1], (1152,))) / 30))[0]
        pairs = [tuple(pair) for pair in row["endpoint_ids"]] + CONTROLS
        for j, (left, right) in enumerate(pairs):
            target[j] = (scores[left] - scores[right]).cpu()
        count += 1

    try:
        for i in range(16):
            forward(i, native[i])
        assert len(a8inputs) == 16
        context["arm"] = "full"
        for i in range(16):
            context["i"] = i
            forward(i, full[i])
        context["arm"] = "single"
        for i, row in enumerate(contexts):
            context["i"] = i
            for destination in range(len(row["destination_positions"])):
                context["destination"] = destination
                forward(i, singles[i][destination])
    finally:
        for hook in hooks:
            hook.remove()
    assert count == 256 and bool(torch.isfinite(native).all()) and bool(torch.isfinite(full).all())
    full_effect = full - native
    single_effect = [value - native[i] for i, value in enumerate(singles)]

    parent_map = {}
    for j, row in enumerate(parent_rows):
        parent_map[(row["family"], row["pair"], row["cue"], row["endpoint"])] = j
    anchor_errors = []
    for i, row in enumerate(contexts):
        for endpoint in range(6):
            j = parent_map[(row["family"], row["pair"], row["cue"], endpoint)]
            anchor_errors.extend([abs(float(native[i, endpoint] - parent[0, j, 0])), abs(float(full[i, endpoint] - parent[1, j, 0]))])
        j = parent_map[(row["family"], row["pair"], row["cue"], 0)]
        anchor_errors.extend((native[i, 6:] - parent[0, j, 2:6]).abs().tolist())
        anchor_errors.extend((full[i, 6:] - parent[1, j, 2:6]).abs().tolist())
    delta_sum_errors = [rel(torch.stack(single_deltas[i]).sum(0), full_deltas[i]) for i in range(16)]
    min_single_delta = min(float(delta.norm()) for values in single_deltas for delta in values)

    paired_targets = {}
    for i in range(0, 16, 2):
        assert contexts[i]["cue"] == "American" and contexts[i + 1]["cue"] == "British"
        key = (contexts[i]["family"], contexts[i]["pair"])
        paired_targets[key] = single_effect[i + 1][:, :6] - single_effect[i][:, :6]
    discovery = torch.cat([v for (family, _), v in paired_targets.items() if family < 2], dim=0)
    _, singular, vh = torch.linalg.svd(discovery, full_matrices=False)
    direction = vh[0]
    discovery_energy = float(singular[0].square() / singular.square().sum().clamp_min(1e-16))
    heldout = []
    composition = []
    selectivity = []
    for family in range(4):
        family_pairs = [paired_targets[(family, pair)] for pair in (0, 1)]
        matrix = torch.cat(family_pairs, dim=0)
        if family >= 2:
            reconstruction = (matrix @ direction)[:, None] * direction[None]
            heldout.append({
                "family": family, "cosine": cos(reconstruction, matrix),
                "relative_reconstruction_error": rel(reconstruction, matrix),
            })
        full_pair = []
        single_sum = []
        for i in range(0, 16, 2):
            if contexts[i]["family"] != family:
                continue
            full_pair.append(full_effect[i + 1, :6] - full_effect[i, :6])
            single_sum.append(paired_targets[(family, contexts[i]["pair"])].sum(0))
        full_pair, single_sum = torch.stack(full_pair), torch.stack(single_sum)
        composition.append({"family": family, "relative_sum_error": rel(single_sum, full_pair)})
        indices = [i for i, row in enumerate(contexts) if row["family"] == family]
        single_rows = torch.cat([single_effect[i] for i in indices], dim=0)
        target_rms = float(single_rows[:, :6].square().mean().sqrt())
        single_control = [float(single_rows[:, 6 + j].square().mean().sqrt() / max(target_rms, 1e-8)) for j in range(4)]
        sum_rows = torch.stack([single_effect[i].sum(0) for i in indices])
        sum_target_rms = float(sum_rows[:, :6].square().mean().sqrt())
        sum_control = [float(sum_rows[:, 6 + j].square().mean().sqrt() / max(sum_target_rms, 1e-8)) for j in range(4)]
        selectivity.append({
            "family": family, "single_control_ratios": single_control,
            "summed_control_ratios": sum_control,
            "max_control_ratio": max(single_control + sum_control),
        })
    torch.save({
        "native": native, "full": full, "singles": singles,
        "discovery_direction": direction, "discovery_singular_values": singular,
    }, artifact)
    result = {
        "pred_a": max(anchor_errors) <= 1e-5 and max(delta_sum_errors) <= 1e-5 and min_single_delta >= 1e-8 and count == 256,
        "pred_b": discovery_energy >= .95,
        "pred_c": all(x["cosine"] >= .9 and x["relative_reconstruction_error"] <= .35 for x in heldout),
        "pred_d": all(x["relative_sum_error"] <= .1 for x in composition),
        "pred_e": all(x["max_control_ratio"] <= .5 for x in selectivity),
        "max_parent_score_anchor_error": max(anchor_errors),
        "max_single_delta_sum_error": max(delta_sum_errors),
        "min_single_destination_delta_norm": min_single_delta,
        "discovery_rank1_energy_fraction": discovery_energy,
        "discovery_singular_values": singular.tolist(),
        "heldout": heldout, "composition": composition, "selectivity": selectivity,
        "body_forwards": count, "seconds": time.perf_counter() - start,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "scope": "Operational response quotient for individual framing destinations of the conditional head8.2 inherited-city to head9.8-O edge. Families0-1 discovery; families2-3 frozen heldout. Native generators and suffix retained.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
