#!/usr/bin/env python3
# BQGATE:576bodyforwards;144prefixes;220seconds;no fitting.
"""pred_a exact descendant-preserving instrument; pred_b capability.
pred_c midpoint law; pred_d pair-centered necessity; pred_e selectivity.
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

STEM = "ODD_ATTENTION8H2_CORPUS_MIDPOINT_REMOVAL_V1"
ARMS = ["native", "full_donor", "midpoint", "self"]
READOUTS = [
    ("target", None), ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
]


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(path) == value for path, value in binding.items())
    rows = json.loads((P / "ODD_ATTENTION8H2_CORPUS_TRANSFER_V1_ROWS.json").read_text())["rows"]
    prior = torch.load(
        P / "ODD_ATTENTION8H2_CORPUS_TRANSFER_V1_ARTIFACT.pt",
        map_location="cpu", weights_only=True,
    )["values"]
    combined = torch.load(
        P / "ODD_ATTENTION8H2_O_COMPOSED_EDGE_V1_PROGRAM.pt",
        map_location="cpu", weights_only=True,
    )
    assert len(rows) == 144 and prior.shape == (5, 144, 5)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        control = json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())
        assert control["pred_a"] and control["rows"] == 144
        print("576bodyforwards;native/full/midpoint/self;CPUcontrol", cpu_control())
        return
    out, artifact = P / (STEM + "_RESULT.json"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not artifact.exists()
    start = time.perf_counter()
    signal.alarm(220)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    edge = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in combined["edge"].items()}
    graph = SharedGraph({k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in combined["o_graph"].items()})
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
        row = rows[i]
        city = row["city_position"]
        ids0 = torch.tensor([row["ids"]], device="cuda")
        idsd = torch.tensor([rows[donor]["ids"]], device="cuda")
        recipient = F.rms_norm(model.transformer.wte(ids0), (1152,))[:, city]
        donor_embedding = F.rms_norm(model.transformer.wte(idsd), (1152,))[:, city]
        target = {1: donor_embedding, 2: (recipient + donor_embedding) / 2, 3: recipient}[arm]
        destination = torch.zeros(len(row["ids"]), dtype=torch.bool, device="cuda")
        destination[row["destination_positions"]] = True
        delta = composed_o_delta(
            edge, graph, a8inputs[i].to("cuda"), city, recipient, target,
            x, x0, first, module.lambdas, destination,
        )
        context["delta"] = delta
        delta_norms[arm].append(float(delta.norm()))
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
    values = torch.zeros(4, 144, 5, dtype=torch.float64)
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
        for i in range(144):
            forward(i, 0)
        assert len(a8inputs) == 144
        for arm in range(1, 4):
            context["arm"] = arm
            for i in range(144):
                context["i"] = i
                forward(i, arm)
    finally:
        for hook in hooks:
            hook.remove()
    assert count == 576 and bool(torch.isfinite(values).all())
    effects = values - values[0]
    anchor = max(
        float((values[0] - prior[0]).abs().max()),
        float((values[1] - prior[4]).abs().max()),
    )
    self_score = float((values[3] - values[0]).abs().max())
    cells = []
    for corpus in ("fineweb", "pile"):
        ix = [i for i, row in enumerate(rows) if row["corpus"] == corpus]
        british, american = ix[::2], ix[1::2]
        native = values[0, british, 0] - values[0, american, 0]
        paired = effects[:, british, 0] - effects[:, american, 0]
        full, midpoint = paired[1], paired[2]
        capable = native > 0
        capable_count = int(capable.sum())
        attenuating = int((midpoint[capable] < 0).sum()) if bool(capable.any()) else 0
        target_rms = float(effects[2, ix, 0].square().mean().sqrt())
        controls = {
            READOUTS[j][0]: float(effects[2, ix, j].square().mean().sqrt() / max(target_rms, 1e-8))
            for j in range(1, 5)
        }
        half_error = float((midpoint - .5 * full).norm() / full.norm().clamp_min(1e-8))
        attenuation_fraction = float(midpoint.abs().mean() / native.abs().mean().clamp_min(1e-8))
        cell = {
            "corpus": corpus, "pairs": len(british),
            "native_positive_pairs": capable_count,
            "full_target_rms": float(full.square().mean().sqrt()),
            "midpoint_half_effect_error": half_error,
            "midpoint_attenuating_capable_pairs": attenuating,
            "midpoint_attenuating_fraction": attenuating / max(capable_count, 1),
            "midpoint_meanabs_over_native_cue": attenuation_fraction,
            "midpoint_target_row_rms": target_rms,
            "midpoint_control_ratios": controls,
        }
        cell["capability_pass"] = capable_count >= len(british) / 2 and cell["full_target_rms"] >= 1e-5
        cell["half_scaling_pass"] = half_error <= .1
        cell["necessity_pass"] = cell["midpoint_attenuating_fraction"] >= .75 and attenuation_fraction >= .02
        cell["controls_pass"] = max(controls.values()) <= .5
        cells.append(cell)
    torch.save({"values": values, "effects": effects}, artifact)
    result = {
        "pred_a": (
            anchor <= 1e-5 and self_score <= 1e-5
            and max(delta_norms[3]) <= 1e-8
            and min(delta_norms[1]) >= 1e-8 and min(delta_norms[2]) >= 1e-8
            and count == 576 and bool(torch.isfinite(values).all())
        ),
        "pred_b": all(c["capability_pass"] for c in cells),
        "pred_c": all(c["half_scaling_pass"] for c in cells),
        "pred_d": all(c["necessity_pass"] for c in cells),
        "pred_e": all(c["controls_pass"] for c in cells),
        "max_parent_anchor_error": anchor,
        "max_self_score_error": self_score,
        "max_self_delta_norm": max(delta_norms[3]),
        "min_full_delta_norm": min(delta_norms[1]),
        "min_midpoint_delta_norm": min(delta_norms[2]),
        "corpora": cells, "body_forwards": count,
        "seconds": time.perf_counter() - start,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "arm_order": ARMS, "readout_order": [name for name, _ in READOUTS],
        "scope": "Pair-centered midpoint removal of the conditional inherited-city head8.2 to head9.8-O edge on cached FineWeb and Pile/reference fragments. Other paths and module services remain native.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
