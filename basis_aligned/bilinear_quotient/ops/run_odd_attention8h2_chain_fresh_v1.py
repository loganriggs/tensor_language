#!/usr/bin/env python3
# BQGATE:288bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact instrument; pred_b capability/live; pred_c value over routing.
pred_d inherited over current; pred_e complete-chain prediction; pred_f controls.
"""
import json, os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F

from attention8h2_city_key_value_v1 import factorial_channels
from attention8h2_city_value_source_v1 import value_arms
from odd_semantic_positions_v1 import semantic_masks
from odd_contextual_positions_v1 import contextual_masks
from odd_source_positions_v1 import select_sources
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from regional_cue_row_check_v1 import validate
from run_even_value_factorial_native_v1 import setup, cpu_control
from run_odd_attention8h2_city_key_value_v1 import head_factor_parts
from sparse_path_stability_atlas_v1 import digest

STEM = "ODD_ATTENTION8H2_CHAIN_FRESH_V1"
ARMS = ["native", "full_city", "routing_only", "value_only", "inherited_only", "current_only"]
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
    assert len(rows) == 48
    validate(rows)
    sem, ctx = semantic_masks(rows), contextual_masks(rows)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        assert json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())["pred_a"]
        print("288bodyforwards;native/full/routing/value/inherited/current;CPUcontrol", cpu_control())
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
    source_replays, factor_errors, value_errors = [], [], []
    current_norms, inherited_norms, outside = [], [], []
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
        framing = sem[i]["framing"].to(x.device)
        city = ctx[i]["city"].to(x.device)
        xr, fr = [z.to(x.device) for z in a8inputs[i]]
        xd, fd = [z.to(x.device) for z in a8inputs[donor]]
        keyd = xr.clone()
        keyd[:, city] = xd[:, city]
        r0, v0 = head_factor_parts(attn8, xr, xr, xr, fr)
        rd, _ = head_factor_parts(attn8, xr, keyd, xr, fr)
        bsz, tokens, _ = xr.shape
        current0 = attn8.c_v(xr).view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        currentd = attn8.c_v(xd).view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        first0 = fr.view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        firstd = fd.view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        vals = value_arms(attn8.lamb, current0, currentd, first0, firstd)
        factors = factorial_channels(r0, rd, vals["recipient"][:, None], vals["donor"][:, None])
        original_channels = factors["recipient"]
        source_replays.append(float((original_channels.sum(-2) - preproj[i].to(x.device)[..., 2 * 128:3 * 128]).norm() / original_channels.sum(-2).norm().clamp_min(1e-8)))
        factor_errors.append(float((factors["additive"] + factors["mixed"] - factors["donor"]).norm() / factors["donor"].norm().clamp_min(1e-8)))
        delta_current = vals["current"] - vals["recipient"]
        delta_inherited = vals["first"] - vals["recipient"]
        delta_full = vals["donor"] - vals["recipient"]
        value_errors.append(float((delta_current + delta_inherited - delta_full).norm() / delta_full.norm().clamp_min(1e-8)))
        current_norms.append(float(delta_current[:, city].norm()))
        inherited_norms.append(float(delta_inherited[:, city].norm()))
        channels = {
            1: factors["donor"], 2: factors["routing"], 3: factors["value"],
            4: r0[..., None] * vals["first"][:, None],
            5: r0[..., None] * vals["current"][:, None],
        }[arm]
        write_delta = F.linear(select_sources(channels - original_channels, city[None]), weight)
        hybrid = x.clone()
        hybrid[:, framing] += write_delta[:, framing]
        outside.append(float((hybrid[:, ~framing] - x[:, ~framing]).abs().max()))
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
        mask = sem[context["i"]]["framing"][None].to(current.device)
        routing, _ = source_factors(graph, current, current, current, first)
        original, _ = value_parts(graph, current, first)
        changed, _ = value_parts(graph, context["hybrid_current"], first)
        delta = select_sources(routing * (changed - original), mask) @ graph.p["output"].double().T
        return output[0] + delta.to(output[0].dtype), output[1]

    hooks.append(model.transformer.h[9].attn.register_forward_hook(a9out))
    values = torch.zeros(6, 48, 6, dtype=torch.float64)
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
        for i in range(48):
            forward(i, 0)
        assert len(a8inputs) == len(preproj) == 48
        for arm in range(1, 6):
            context["arm"] = arm
            for i in range(48):
                context["i"] = i
                forward(i, arm)
    finally:
        for hook in hooks:
            hook.remove()
    assert count == 288 and bool(torch.isfinite(values).all())
    effect = values - values[0]
    cells = []
    for family in range(2):
        ix = [i for i, row in enumerate(rows) if row["family"] == family]
        native = values[0, ix[::2], 0] - values[0, ix[1::2], 0]
        cue = effect[:, ix[::2], 0] - effect[:, ix[1::2], 0]
        full, value = cue[1], cue[3]
        target = float(full.square().mean().sqrt())
        rel_full = lambda arm: float((cue[arm] - full).norm() / full.norm().clamp_min(1e-8))
        rel_value = lambda arm: float((cue[arm] - value).norm() / value.norm().clamp_min(1e-8))
        routing_error, value_error = rel_full(2), rel_full(3)
        inherited_value_error, current_value_error, inherited_full_error = rel_value(4), rel_value(5), rel_full(4)
        controls = {ARMS[arm]: {READOUTS[j][0]: float(effect[arm, ix, j].square().mean().sqrt() / max(target, 1e-8)) for j in range(2, 6)} for arm in range(1, 6)}
        max_control = max(x for arm in controls.values() for x in arm.values())
        cells.append({
            "family": family, "native_positive_pairs": int((native > 0).sum()),
            "native_cue_mean": float(native.mean()), "full_city_target_rms": target,
            "routing_to_full_error": routing_error, "value_to_full_error": value_error,
            "inherited_to_value_error": inherited_value_error, "current_to_value_error": current_value_error,
            "inherited_to_full_error": inherited_full_error, "control_ratios_to_full_target": controls,
            "max_control_ratio": max_control,
            "capability_pass": int((native > 0).sum()) >= 10 and target >= 1e-5,
            "value_transport_pass": value_error <= .35 and routing_error >= .5,
            "inherited_provenance_pass": inherited_value_error <= .35 and current_value_error >= .5,
            "complete_chain_pass": inherited_full_error <= .35, "controls_pass": max_control <= .5,
        })
    torch.save({"values": values, "effects": effect}, artifact)
    result = {
        "pred_a": max(source_replays) <= 1e-5 and max(factor_errors) <= 1e-5 and max(value_errors) <= 1e-5 and min(current_norms) >= 1e-8 and min(inherited_norms) >= 1e-8 and max(outside) == 0 and count == 288 and bool(torch.isfinite(values).all()),
        "pred_b": all(c["capability_pass"] for c in cells),
        "pred_c": all(c["value_transport_pass"] for c in cells),
        "pred_d": all(c["inherited_provenance_pass"] for c in cells),
        "pred_e": all(c["complete_chain_pass"] for c in cells),
        "pred_f": all(c["controls_pass"] for c in cells),
        "max_head_source_replay_error": max(source_replays), "max_factorial_error": max(factor_errors),
        "max_value_sum_error": max(value_errors), "min_current_city_delta_norm": min(current_norms),
        "min_inherited_city_delta_norm": min(inherited_norms), "max_outside_hybrid_delta": max(outside),
        "families": cells, "body_forwards": count, "seconds": time.perf_counter() - start,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "arm_order": ARMS, "readout_order": [x[0] for x in READOUTS],
        "scope": "Frozen inherited-city head8.2 framing-write to head9.8-O chain on unseen templates, cities and endpoints. Authored task shift, not corpus OOD or whole-head/full-model sufficiency.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
