#!/usr/bin/env python3
# BQGATE:720bodyforwards;144prefixes;240seconds;no fitting.
"""pred_a instrument; pred_b corpus capability; pred_c transported computation.
pred_d signed manipulation; pred_e matched unrelated-token selectivity.
"""
import hashlib
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

from attention8h2_city_key_value_v1 import factorial_channels
from attention8h2_city_value_source_v1 import value_arms
from odd_source_positions_v1 import select_sources
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from run_even_value_factorial_native_v1 import setup, cpu_control
from run_odd_attention8h2_city_key_value_v1 import head_factor_parts
from sparse_path_stability_atlas_v1 import digest

STEM = "ODD_ATTENTION8H2_CORPUS_TRANSFER_V1"
ARMS = ["native", "full_city", "routing_only", "value_only", "inherited_edge"]
READOUTS = [
    ("target", None), ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
]


def row_hash(row):
    copy = dict(row)
    claimed = copy.pop("row_sha256")
    payload = json.dumps(copy, sort_keys=True, separators=(",", ":")).encode()
    return claimed, hashlib.sha256(payload).hexdigest()


def relative(x, y):
    return float((x - y).norm() / y.norm().clamp_min(1e-8))


def cosine(x, y):
    return float((x @ y) / (x.norm() * y.norm()).clamp_min(1e-8))


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(k) == v for k, v in binding.items())
    rows_doc = json.loads((P / (STEM + "_ROWS.json")).read_text())
    rows = rows_doc["rows"]
    assert len(rows) == 144
    assert all(row_hash(row)[0] == row_hash(row)[1] for row in rows)
    assert all(
        rows[i]["cue"] == "British" and rows[i + 1]["cue"] == "American"
        and rows[i]["context_id"] == rows[i + 1]["context_id"]
        and rows[i]["endpoint"] == rows[i + 1]["endpoint"]
        and sum(a != b for a, b in zip(rows[i]["ids"], rows[i + 1]["ids"])) == 1
        for i in range(0, len(rows), 2)
    )
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        control = json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())
        assert control["pred_a"] and control["rows"] == 144
        print("720bodyforwards;native/full/routing/value/inherited;CPUcontrol", cpu_control())
        return
    out, artifact = P / (STEM + "_RESULT.json"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not artifact.exists()
    start = time.perf_counter()
    signal.alarm(240)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    graph, _, _, _ = setup("cuda")
    context = {"arm": 0}
    a8inputs, preproj = [], []
    source_replays, factor_errors, value_errors = [], [], []
    arm_write_norms, outside = {a: [] for a in range(1, 5)}, []
    attn8 = model.transformer.h[8].attn
    weight = attn8.c_proj.weight[:, 2 * 128:3 * 128]

    def a8pre(module, args):
        if context["arm"] == 0:
            a8inputs.append((args[0].detach().cpu(), args[1].detach().cpu()))

    def cpre(module, args):
        if context["arm"] == 0:
            preproj.append(args[0].detach().cpu())

    def masks(row, device):
        city = torch.zeros(len(row["ids"]), dtype=torch.bool, device=device)
        city[row["city_position"]] = True
        destination = torch.zeros_like(city)
        destination[row["destination_positions"]] = True
        return city, destination

    def b9pre(module, args):
        x, v1, x0 = args
        if context["arm"] == 0:
            return None
        i, arm = context["i"], context["arm"]
        donor = i ^ 1
        city, destination = masks(rows[i], x.device)
        xr, fr = [z.to(x.device) for z in a8inputs[i]]
        xd, fd = [z.to(x.device) for z in a8inputs[donor]]
        keyd = xr.clone()
        keyd[:, city] = xd[:, city]
        r0, _ = head_factor_parts(attn8, xr, xr, xr, fr)
        rd, _ = head_factor_parts(attn8, xr, keyd, xr, fr)
        bsz, tokens, _ = xr.shape
        current0 = attn8.c_v(xr).view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        currentd = attn8.c_v(xd).view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        first0 = fr.view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        firstd = fd.view(bsz, tokens, attn8.n_head, attn8.head_dim)[:, :, 2]
        vals = value_arms(attn8.lamb, current0, currentd, first0, firstd)
        factors = factorial_channels(r0, rd, vals["recipient"][:, None], vals["donor"][:, None])
        original = factors["recipient"]
        source_replays.append(float(
            (original.sum(-2) - preproj[i].to(x.device)[..., 2 * 128:3 * 128]).norm()
            / original.sum(-2).norm().clamp_min(1e-8)
        ))
        factor_errors.append(float(
            (factors["additive"] + factors["mixed"] - factors["donor"]).norm()
            / factors["donor"].norm().clamp_min(1e-8)
        ))
        value_errors.append(float(
            ((vals["current"] - vals["recipient"]) + (vals["first"] - vals["recipient"])
             - (vals["donor"] - vals["recipient"])).norm()
            / (vals["donor"] - vals["recipient"]).norm().clamp_min(1e-8)
        ))
        channels = {
            1: factors["donor"],
            2: factors["routing"],
            3: factors["value"],
            4: r0[..., None] * vals["first"][:, None],
        }[arm]
        write_delta = F.linear(select_sources(channels - original, city[None]), weight)
        arm_write_norms[arm].append(float(write_delta[:, destination].norm()))
        hybrid = x.clone()
        hybrid[:, destination] += write_delta[:, destination]
        outside.append(float((hybrid[:, ~destination] - x[:, ~destination]).abs().max()))
        mixed = module.lambdas[0] * hybrid + module.lambdas[1] * x0
        context["hybrid_current"] = F.rms_norm(mixed, (mixed.size(-1),))
        return None

    hooks = [
        attn8.register_forward_pre_hook(a8pre),
        attn8.c_proj.register_forward_pre_hook(cpre),
        model.transformer.h[9].register_forward_pre_hook(b9pre),
    ]

    def a9out(module, args, output):
        if context["arm"] == 0:
            return output
        current, first = args
        _, destination = masks(rows[context["i"]], current.device)
        routing, _ = source_factors(graph, current, current, current, first)
        original, _ = value_parts(graph, current, first)
        changed, _ = value_parts(graph, context["hybrid_current"], first)
        delta = select_sources(
            routing * (changed - original), destination[None]
        ) @ graph.p["output"].double().T
        return output[0] + delta.to(output[0].dtype), output[1]

    hooks.append(model.transformer.h[9].attn.register_forward_hook(a9out))
    values = torch.zeros(5, 144, 5, dtype=torch.float64)
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
        for i in range(144):
            forward(i, 0)
        assert len(a8inputs) == len(preproj) == 144
        for arm in range(1, 5):
            context["arm"] = arm
            for i in range(144):
                context["i"] = i
                forward(i, arm)
    finally:
        for hook in hooks:
            hook.remove()
    assert count == 720 and bool(torch.isfinite(values).all())
    effects = values - values[0]
    cells = []
    for corpus in ("fineweb", "pile"):
        ix = [i for i, row in enumerate(rows) if row["corpus"] == corpus]
        british, american = ix[::2], ix[1::2]
        native = values[0, british, 0] - values[0, american, 0]
        paired = effects[:, british, 0] - effects[:, american, 0]
        full, inherited = paired[1], paired[4]
        capable = native > 0
        full_rms = float(full.square().mean().sqrt())
        inherited_row_target = float(effects[4, ix, 0].square().mean().sqrt())
        controls = {
            READOUTS[j][0]: float(
                effects[4, ix, j].square().mean().sqrt() / max(inherited_row_target, 1e-8)
            ) for j in range(1, 5)
        }
        donor_directed = int((inherited[capable] < 0).sum()) if bool(capable.any()) else 0
        capable_count = int(capable.sum())
        donor_fraction = donor_directed / max(capable_count, 1)
        magnitude_fraction = float(inherited.abs().mean() / native.abs().mean().clamp_min(1e-8))
        cell = {
            "corpus": corpus, "rows": len(ix), "pairs": len(british),
            "native_positive_pairs": capable_count,
            "native_cue_mean": float(native.mean()),
            "full_city_effect_rms": full_rms,
            "value_to_full_error": relative(paired[3], full),
            "routing_to_full_error": relative(paired[2], full),
            "inherited_to_full_error": relative(inherited, full),
            "inherited_full_cosine": cosine(inherited, full),
            "donor_directed_capable_pairs": donor_directed,
            "donor_directed_fraction": donor_fraction,
            "inherited_meanabs_over_native_cue": magnitude_fraction,
            "inherited_target_row_rms": inherited_row_target,
            "control_ratios": controls,
        }
        cell["capability_pass"] = capable_count >= len(british) / 2 and full_rms >= 1e-5
        cell["transport_pass"] = (
            cell["value_to_full_error"] <= .35
            and cell["routing_to_full_error"] >= .5
            and cell["inherited_to_full_error"] <= .35
        )
        cell["manipulation_pass"] = (
            donor_fraction >= .75 and cell["inherited_full_cosine"] >= .6
            and magnitude_fraction >= .02
        )
        cell["controls_pass"] = max(controls.values()) <= .5
        cells.append(cell)
    torch.save({"values": values, "effects": effects}, artifact)
    result = {
        "pred_a": (
            max(source_replays) <= 1e-5 and max(factor_errors) <= 1e-5
            and max(value_errors) <= 1e-5 and max(outside) == 0
            and all(min(v) >= 1e-8 for v in arm_write_norms.values())
            and count == 720 and bool(torch.isfinite(values).all())
        ),
        "pred_b": all(c["capability_pass"] for c in cells),
        "pred_c": all(c["transport_pass"] for c in cells),
        "pred_d": all(c["manipulation_pass"] for c in cells),
        "pred_e": all(c["controls_pass"] for c in cells),
        "max_head_source_replay_error": max(source_replays),
        "max_factorial_error": max(factor_errors),
        "max_value_sum_error": max(value_errors),
        "min_arm_write_norms": {ARMS[k]: min(v) for k, v in arm_write_norms.items()},
        "max_outside_hybrid_delta": max(outside),
        "corpora": cells, "body_forwards": count,
        "seconds": time.perf_counter() - start,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "arm_order": ARMS, "readout_order": [x[0] for x in READOUTS],
        "scope": "Outcome-blind cached FineWeb and Pile/reference fragments with one natural arm and one city substitution. No post-score filtering, verified pretraining-disjointness, whole-head removal, or isolated compression.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
