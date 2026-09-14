#!/usr/bin/env python3
# BQGATE:384bodyforwards;96prefixes;180seconds;no fitting.
"""pred_a exact anchors/partition; pred_b capability/live.
pred_c description destinations; pred_d instruction; pred_e compose; pred_f controls.
"""
import json, os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F

from attention8h2_city_value_source_v1 import value_arms
from odd_attention8h2_destination_role_v1 import destination_parts
from odd_contextual_positions_v1 import contextual_masks
from odd_framing_role_split_v1 import role_masks
from odd_source_positions_v1 import select_sources
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from regional_cue_row_check_v1 import validate
from run_even_value_factorial_native_v1 import setup, cpu_control
from run_odd_attention8h2_city_key_value_v1 import head_factor_parts
from sparse_path_stability_atlas_v1 import digest

STEM = "ODD_ATTENTION8H2_DESTINATION_ROLE_V1"
ARMS = ["native", "full_inherited", "description", "instruction"]
READOUTS = [
    ("target", None), ("work_jobs", (670, 3946)), ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
]


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(k) == v for k, v in binding.items())
    rows = json.loads((P / (STEM + "_ROWS.json")).read_text())["rows"]
    assert len(rows) == 96
    validate(rows)
    roles, contexts = role_masks(rows), contextual_masks(rows)
    old = torch.load(P / "ODD_ATTENTION8H2_CITY_VALUE_SOURCE_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)["values"]
    fresh = torch.load(P / "ODD_ATTENTION8H2_CHAIN_FRESH_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)["values"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        assert json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())["pred_a"]
        print("384bodyforwards;native/fullinherited/description/instruction;CPUcontrol", cpu_control())
        return
    out, artifact = P / (STEM + "_RESULT.json"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not artifact.exists()
    start = time.perf_counter()
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    graph, _, _, _ = setup("cuda")
    context = {"arm": 0}
    a8inputs, preproj = [], []
    source_replays, partitions, outside = [], [], []
    attn8 = model.transformer.h[8].attn
    weight = attn8.c_proj.weight[:, 2 * 128:3 * 128]

    def a8pre(module, args):
        if context["arm"] == 0:
            a8inputs.append((args[0].detach().cpu(), args[1].detach().cpu()))

    def cpre(module, args):
        if context["arm"] == 0:
            preproj.append(args[0].detach().cpu())

    def b9pre(module, args):
        x, v1, x0 = args
        if context["arm"] == 0:
            return None
        i, arm = context["i"], context["arm"]
        donor = i ^ 1
        city = contexts[i]["city"].to(x.device)
        xr, fr = [z.to(x.device) for z in a8inputs[i]]
        xd, fd = [z.to(x.device) for z in a8inputs[donor]]
        routing, v0 = head_factor_parts(attn8, xr, xr, xr, fr)
        original_channels = routing[..., None] * v0[:, None]
        source_replays.append(float((original_channels.sum(-2) - preproj[i].to(x.device)[..., 2 * 128:3 * 128]).norm() / original_channels.sum(-2).norm().clamp_min(1e-8)))
        bsz, tokens, _ = xr.shape
        current0 = attn8.c_v(xr).view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        currentd = attn8.c_v(xd).view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        first0 = fr.view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        firstd = fd.view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        inherited = value_arms(attn8.lamb, current0, currentd, first0, firstd)["first"]
        changed_channels = routing[..., None] * inherited[:, None]
        write_delta = F.linear(select_sources(changed_channels - original_channels, city[None]), weight)
        parts = destination_parts(write_delta, roles[i])
        denom = parts["framing"].norm().clamp_min(1e-8)
        partitions.append(float((parts["description"] + parts["instruction"] - parts["framing"]).norm() / denom))
        chosen = {1: "framing", 2: "description", 3: "instruction"}[arm]
        hybrid = x + parts[chosen]
        outside.append(float((parts[chosen][:, ~roles[i][chosen]] if chosen != "framing" else parts[chosen][:, ~roles[i]["framing"]]).abs().max()))
        mixed = module.lambdas[0] * hybrid + module.lambdas[1] * x0
        context["hybrid_current"] = F.rms_norm(mixed, (mixed.size(-1),))
        return None

    hooks = [
        attn8.register_forward_pre_hook(a8pre), attn8.c_proj.register_forward_pre_hook(cpre),
        model.transformer.h[9].register_forward_pre_hook(b9pre),
    ]

    def a9out(module, args, output):
        if context["arm"] == 0:
            return output
        current, first = args
        mask = roles[context["i"]]["framing"][None].to(current.device)
        routing, _ = source_factors(graph, current, current, current, first)
        original, _ = value_parts(graph, current, first)
        changed, _ = value_parts(graph, context["hybrid_current"], first)
        delta = select_sources(routing * (changed - original), mask) @ graph.p["output"].double().T
        return output[0] + delta.to(output[0].dtype), output[1]

    hooks.append(model.transformer.h[9].attn.register_forward_hook(a9out))
    values = torch.zeros(4, 96, 6, dtype=torch.float64)
    count = 0

    def forward(i, arm):
        nonlocal count
        row = rows[i]
        ids = torch.tensor([row["ids"]], device="cuda")
        x = F.rms_norm(model.transformer.wte(ids), (1152,))
        x0, v1 = x, None
        for block in model.transformer.h:
            x, v1 = block(x, v1, x0)
        scores = (30 * torch.tanh(model.lm_head(F.rms_norm(x[:, -1], (1152,))) / 30))[0]
        pairs = [(row["uk_id"], row["us_id"])] + [pair for _, pair in READOUTS[1:]]
        for j, (left, right) in enumerate(pairs):
            values[arm, i, j] = (scores[left] - scores[right]).cpu()
        count += 1

    try:
        for i in range(96):
            forward(i, 0)
        assert len(a8inputs) == len(preproj) == 96
        for arm in range(1, 4):
            context["arm"] = arm
            for i in range(96):
                context["i"] = i
                forward(i, arm)
    finally:
        for hook in hooks:
            hook.remove()
    assert count == 384 and bool(torch.isfinite(values).all())
    anchor = max(
        float((values[0, :48, :2] - old[0, :, :2]).abs().max()),
        float((values[1, :48, :2] - old[3, :, :2]).abs().max()),
        float((values[0, 48:, :2] - fresh[0, :, :2]).abs().max()),
        float((values[1, 48:, :2] - fresh[4, :, :2]).abs().max()),
    )
    effect = values - values[0]
    cells = []
    for family in range(4):
        ix = [i for i, row in enumerate(rows) if row["family"] == family]
        native = values[0, ix[::2], 0] - values[0, ix[1::2], 0]
        cue = effect[:, ix[::2], 0] - effect[:, ix[1::2], 0]
        full = cue[1]
        target = float(full.square().mean().sqrt())
        rel = lambda arm: float((cue[arm] - full).norm() / full.norm().clamp_min(1e-8))
        description_error, instruction_error = rel(2), rel(3)
        composition = float((cue[2] + cue[3] - full).norm() / full.norm().clamp_min(1e-8))
        controls = {ARMS[arm]: {READOUTS[j][0]: float(effect[arm, ix, j].square().mean().sqrt() / max(target, 1e-8)) for j in range(2, 6)} for arm in range(1, 4)}
        max_control = max(x for arm in controls.values() for x in arm.values())
        cells.append({
            "family": family, "native_positive_pairs": int((native > 0).sum()),
            "native_cue_mean": float(native.mean()), "full_inherited_target_rms": target,
            "description_to_full_error": description_error, "instruction_to_full_error": instruction_error,
            "separate_effect_composition_error": composition, "control_ratios_to_full_target": controls,
            "max_control_ratio": max_control,
            "capability_pass": int((native > 0).sum()) >= 10 and target >= 1e-5,
            "description_hypothesis_pass": description_error <= .35 and instruction_error >= .5,
            "instruction_hypothesis_pass": instruction_error <= .35 and description_error >= .5,
            "composition_pass": composition <= .1, "controls_pass": max_control <= .5,
        })
    torch.save({"values": values, "effects": effect}, artifact)
    result = {
        "pred_a": anchor <= 1e-5 and max(source_replays) <= 1e-5 and max(partitions) <= 1e-5 and max(outside) == 0 and count == 384 and bool(torch.isfinite(values).all()),
        "pred_b": all(c["capability_pass"] for c in cells),
        "pred_c": all(c["description_hypothesis_pass"] for c in cells),
        "pred_d": all(c["instruction_hypothesis_pass"] for c in cells),
        "pred_e": all(c["composition_pass"] for c in cells),
        "pred_f": all(c["controls_pass"] for c in cells),
        "max_frozen_anchor_error": anchor, "max_head_source_replay_error": max(source_replays),
        "max_destination_partition_error": max(partitions), "max_outside_destination_delta": max(outside),
        "families": cells, "body_forwards": count, "seconds": time.perf_counter() - start,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "arm_order": ARMS, "readout_order": [x[0] for x in READOUTS],
        "scope": "Inherited-city head8.2 write split across token-defined description/instruction destinations, then isolated head9.8-O and native suffix on four frozen templates. No fitted role or corpus OOD.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
