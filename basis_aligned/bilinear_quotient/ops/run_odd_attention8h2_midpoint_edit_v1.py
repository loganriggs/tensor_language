#!/usr/bin/env python3
# BQGATE:384bodyforwards;96prefixes;180seconds;no fitting.
"""pred_a anchors/self; pred_b capability/live; pred_c half scaling; pred_d attenuation; pred_e controls."""
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

STEM = "ODD_ATTENTION8H2_MIDPOINT_EDIT_V1"
ARMS = ["native", "full_donor", "midpoint", "self"]
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
    combined = torch.load(P / "ODD_ATTENTION8H2_O_COMPOSED_EDGE_V1_PROGRAM.pt", map_location="cpu", weights_only=True)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        assert json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())["pred_a"]
        print("384bodyforwards;native/full/midpoint/self;CPUcontrol", cpu_control())
        return
    out, artifact = P / (STEM + "_RESULT.json"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not artifact.exists()
    start = time.perf_counter()
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    edge = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in combined["edge"].items()}
    graph = SharedGraph({key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in combined["o_graph"].items()})
    context = {"arm": 0}
    a8inputs, delta_norms = [], {1: [], 2: [], 3: []}

    def a8pre(module, args):
        if context["arm"] == 0:
            a8inputs.append(args[0].detach().cpu())

    def b9pre(module, args):
        if context["arm"] == 0:
            return None
        i, donor, arm = context["i"], context["i"] ^ 1, context["arm"]
        x, first, x0 = args
        city = int(torch.nonzero(contexts[i]["city"])[0])
        ids0 = torch.tensor([rows[i]["ids"]], device="cuda")
        idsd = torch.tensor([rows[donor]["ids"]], device="cuda")
        recipient = F.rms_norm(model.transformer.wte(ids0), (1152,))[:, city]
        donor_embedding = F.rms_norm(model.transformer.wte(idsd), (1152,))[:, city]
        target = {1: donor_embedding, 2: (recipient + donor_embedding) / 2, 3: recipient}[arm]
        delta = composed_o_delta(edge, graph, a8inputs[i].to("cuda"), city, recipient, target, x, x0, first, module.lambdas, roles[i]["framing"].to("cuda"))
        context["delta"] = delta
        delta_norms[arm].append(float(delta.norm()))
        return None

    def a9out(module, args, output):
        if context["arm"] == 0:
            return output
        return output[0] + context["delta"].to(output[0].dtype), output[1]

    hooks = [model.transformer.h[8].attn.register_forward_pre_hook(a8pre), model.transformer.h[9].register_forward_pre_hook(b9pre), model.transformer.h[9].attn.register_forward_hook(a9out)]
    values = torch.zeros(4, 96, 6, dtype=torch.float64)
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
        for arm in range(1, 4):
            context["arm"] = arm
            for i in range(96):
                context["i"] = i
                forward(i, arm)
    finally:
        for hook in hooks:
            hook.remove()
    effect = values - values[0]
    anchor = max(float((values[0, :, :2] - prior[0, :, :2]).abs().max()), float((values[1, :, :2] - prior[1, :, :2]).abs().max()))
    self_anchor = float((values[3] - values[0]).abs().max())
    cells = []
    for family in range(4):
        ix = [i for i, row in enumerate(rows) if row["family"] == family]
        native = values[0, ix[::2], 0] - values[0, ix[1::2], 0]
        cues = effect[:, ix[::2], 0] - effect[:, ix[1::2], 0]
        full, midpoint = cues[1], cues[2]
        target = float(full.square().mean().sqrt())
        half_error = float((midpoint - .5 * full).norm() / full.norm().clamp_min(1e-8))
        attenuation_fraction = float((-midpoint / native).mean())
        controls = {READOUTS[j][0]: float(effect[2, ix, j].square().mean().sqrt() / max(float(midpoint.square().mean().sqrt()), 1e-8)) for j in range(2, 6)}
        cells.append({
            "family": family, "native_positive_pairs": int((native > 0).sum()), "full_target_rms": target,
            "midpoint_half_effect_error": half_error, "midpoint_attenuating_pairs": int((midpoint < 0).sum()),
            "mean_midpoint_attenuation_fraction": attenuation_fraction, "midpoint_control_ratios": controls,
            "capability_pass": int((native > 0).sum()) >= 10 and target >= 1e-5,
            "half_scaling_pass": half_error <= .1,
            "attenuation_pass": int((midpoint < 0).sum()) == 12 and attenuation_fraction >= .02,
            "controls_pass": max(controls.values()) <= .5,
        })
    torch.save({"values": values, "effects": effect}, artifact)
    result = {
        "pred_a": anchor <= 1e-5 and self_anchor <= 1e-5 and max(delta_norms[3]) <= 1e-8 and count == 384 and bool(torch.isfinite(values).all()),
        "pred_b": all(cell["capability_pass"] for cell in cells),
        "pred_c": all(cell["half_scaling_pass"] for cell in cells),
        "pred_d": all(cell["attenuation_pass"] for cell in cells),
        "pred_e": all(cell["controls_pass"] for cell in cells),
        "max_parent_anchor_error": anchor, "max_self_score_error": self_anchor,
        "max_self_delta_norm": max(delta_norms[3]), "families": cells,
        "body_forwards": count, "seconds": time.perf_counter() - start,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "arm_order": ARMS, "readout_order": [name for name, _ in READOUTS],
        "scope": "Midpoint strength intervention on the conditional head8.2 inherited-city to head9.8-O edge across four frozen templates. No whole-head removal or corpus OOD.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
