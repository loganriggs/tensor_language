#!/usr/bin/env python3
# BQGATE:192bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact value split/anchors; pred_b capability/live.
pred_c current provenance; pred_d inherited provenance; pred_e effects compose.
"""
import json, os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F

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

STEM = "ODD_ATTENTION8H2_CITY_VALUE_SOURCE_V1"
ARMS = ["native", "full_value", "current_only", "inherited_only"]
READOUTS = [
    ("target", None), ("work_jobs", (670, 3946)), ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
]


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(k) == v for k, v in binding.items())
    rows = json.loads((P / "ODD_FRAMING_FRESH_V1_ROWS.json").read_text())["rows"]
    assert len(rows) == 48
    validate(rows)
    sem, ctx = semantic_masks(rows), contextual_masks(rows)
    prior = torch.load(P / "ODD_ATTENTION8H2_CITY_KEY_VALUE_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)["values"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        assert json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())["pred_a"]
        print("192bodyforwards;native/fullvalue/current/inherited;CPUcontrol", cpu_control())
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
    a8inputs, sum_errors, current_norms, inherited_norms, outside = [], [], [], [], []
    attn8 = model.transformer.h[8].attn
    weight = attn8.c_proj.weight[:, 2 * 128:3 * 128]

    def a8pre(module, args):
        if context["arm"] == 0:
            a8inputs.append((args[0].detach().cpu(), args[1].detach().cpu()))

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
        routing, _ = head_factor_parts(attn8, xr, xr, xr, fr)
        bsz, tokens, _ = xr.shape
        current0 = attn8.c_v(xr).view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        currentd = attn8.c_v(xd).view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        first0 = fr.view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        firstd = fd.view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        vals = value_arms(attn8.lamb, current0, currentd, first0, firstd)
        delta_current = vals["current"] - vals["recipient"]
        delta_inherited = vals["first"] - vals["recipient"]
        delta_full = vals["donor"] - vals["recipient"]
        sum_errors.append(float((delta_current + delta_inherited - delta_full).norm() / delta_full.norm().clamp_min(1e-8)))
        current_norms.append(float(delta_current[:, city].norm()))
        inherited_norms.append(float(delta_inherited[:, city].norm()))
        chosen = {1: "donor", 2: "current", 3: "first"}[arm]
        recipient_channel = routing[..., None] * vals["recipient"][:, None]
        changed_channel = routing[..., None] * vals[chosen][:, None]
        write_delta = F.linear(select_sources(changed_channel - recipient_channel, city[None]), weight)
        hybrid = x.clone()
        hybrid[:, framing] += write_delta[:, framing]
        outside.append(float((hybrid[:, ~framing] - x[:, ~framing]).abs().max()))
        mixed = module.lambdas[0] * hybrid + module.lambdas[1] * x0
        context["hybrid_current"] = F.rms_norm(mixed, (mixed.size(-1),))
        return None

    hooks = [attn8.register_forward_pre_hook(a8pre), model.transformer.h[9].register_forward_pre_hook(b9pre)]

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
    values = torch.zeros(4, 48, 6, dtype=torch.float64)
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
        assert len(a8inputs) == 48
        for arm in range(1, 4):
            context["arm"] = arm
            for i in range(48):
                context["i"] = i
                forward(i, arm)
    finally:
        for hook in hooks:
            hook.remove()
    assert count == 192 and bool(torch.isfinite(values).all())
    effect = values - values[0]
    anchor = max(float((values[0, :, :2] - prior[0, :, :2]).abs().max()), float((values[1, :, :2] - prior[3, :, :2]).abs().max()))
    cells = []
    for family in range(2):
        ix = [i for i, row in enumerate(rows) if row["family"] == family]
        native = values[0, ix[::2], 0] - values[0, ix[1::2], 0]
        cue = effect[:, ix[::2], 0] - effect[:, ix[1::2], 0]
        full = cue[1]
        target = float(full.square().mean().sqrt())
        rel = lambda arm: float((cue[arm] - full).norm() / full.norm().clamp_min(1e-8))
        current_error, inherited_error = rel(2), rel(3)
        composition = float((cue[2] + cue[3] - full).norm() / full.norm().clamp_min(1e-8))
        controls = {ARMS[arm]: {READOUTS[j][0]: float(effect[arm, ix, j].square().mean().sqrt() / max(target, 1e-8)) for j in range(2, 6)} for arm in range(1, 4)}
        cells.append({
            "family": family, "native_positive_pairs": int((native > 0).sum()),
            "native_cue_mean": float(native.mean()), "full_value_target_rms": target,
            "current_to_full_cue_error": current_error, "inherited_to_full_cue_error": inherited_error,
            "separate_effect_composition_error": composition, "control_ratios_to_full_target": controls,
            "capability_pass": int((native > 0).sum()) >= 10 and target >= 1e-5,
            "current_hypothesis_pass": current_error <= .35 and inherited_error >= .5,
            "inherited_hypothesis_pass": inherited_error <= .35 and current_error >= .5,
            "composition_pass": composition <= .1,
        })
    torch.save({"values": values, "effects": effect}, artifact)
    result = {
        "pred_a": anchor <= 1e-5 and max(sum_errors) <= 1e-5 and min(current_norms) >= 1e-8 and min(inherited_norms) >= 1e-8 and max(outside) == 0 and count == 192 and bool(torch.isfinite(values).all()),
        "pred_b": all(c["capability_pass"] for c in cells),
        "pred_c": all(c["current_hypothesis_pass"] for c in cells),
        "pred_d": all(c["inherited_hypothesis_pass"] for c in cells),
        "pred_e": all(c["composition_pass"] for c in cells),
        "max_frozen_anchor_error": anchor, "max_value_sum_error": max(sum_errors),
        "min_current_city_delta_norm": min(current_norms), "min_inherited_city_delta_norm": min(inherited_norms),
        "max_outside_hybrid_delta": max(outside), "families": cells,
        "body_forwards": count, "seconds": time.perf_counter() - start,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "arm_order": ARMS, "readout_order": [x[0] for x in READOUTS],
        "scope": "Recipient-routing-fixed head8.2 city value split into current and inherited sources, propagated through framing writes, isolated head9.8-O current values, and native suffix. No whole-head sufficiency, corpus OOD, compression adoption, or quantization.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
