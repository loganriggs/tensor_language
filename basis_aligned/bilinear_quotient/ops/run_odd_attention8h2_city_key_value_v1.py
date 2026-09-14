#!/usr/bin/env python3
# BQGATE:240bodyforwards;48prefixes;180seconds;no fitting.
"""pred_a exact factorial/anchors; pred_b capability/live; pred_c mixed omissible.
pred_d routing transport; pred_e value transport.
"""
import json, os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F

from attention8h2_city_key_value_v1 import factorial_channels
from odd_semantic_positions_v1 import semantic_masks
from odd_contextual_positions_v1 import contextual_masks
from odd_source_positions_v1 import select_sources
from odd_source_swap_interaction_v1 import source_factors
from odd_value_source_split_v1 import value_parts
from regional_cue_row_check_v1 import validate
from run_even_value_factorial_native_v1 import setup, cpu_control
from sparse_path_stability_atlas_v1 import digest

STEM = "ODD_ATTENTION8H2_CITY_KEY_VALUE_V1"
ARMS = ["native", "full_city", "routing_only", "value_only", "additive_no_mixed"]
READOUTS = [
    ("target", None), ("work_jobs", (670, 3946)), ("cat_dog", (3797, 3290)),
    ("red_blue", (2266, 4171)), ("monday_tuesday", (3321, 3431)),
    ("apple_orange", (17180, 10912)),
]


def head_factor_parts(attn, query_current, key_current, value_current, first, head=2):
    from jacclust.tt_model import apply_rotary_emb
    bsz, tokens, _ = query_current.shape
    heads, dim = attn.n_head, attn.head_dim
    q = attn.c_q(query_current).view(bsz, tokens, heads, dim)
    q2 = attn.c_q2(query_current).view(bsz, tokens, heads, dim)
    k = attn.c_k(key_current).view(bsz, tokens, heads, dim)
    k2 = attn.c_k2(key_current).view(bsz, tokens, heads, dim)
    v = attn.c_v(value_current).view(bsz, tokens, heads, dim)
    v = (1 - attn.lamb) * v + attn.lamb * first.view_as(v)
    cos, sin = attn.rotary(q)
    q = apply_rotary_emb(F.rms_norm(q, (dim,)), cos, sin)
    q2 = apply_rotary_emb(F.rms_norm(q2, (dim,)), cos, sin)
    k = apply_rotary_emb(F.rms_norm(k, (dim,)), cos, sin)
    k2 = apply_rotary_emb(F.rms_norm(k2, (dim,)), cos, sin)
    a = torch.einsum("bqd,bkd->bqk", q[:, :, head], k[:, :, head]) / dim
    b = torch.einsum("bqd,bkd->bqk", q2[:, :, head], k2[:, :, head]) / dim
    causal = torch.ones(tokens, tokens, dtype=torch.bool, device=q.device).tril()
    return (a * b).masked_fill(~causal, 0), v[:, :, head, :]


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(k) == v for k, v in binding.items())
    rows = json.loads((P / "ODD_FRAMING_FRESH_V1_ROWS.json").read_text())["rows"]
    assert len(rows) == 48
    validate(rows)
    sem, ctx = semantic_masks(rows), contextual_masks(rows)
    prior = torch.load(P / "ODD_ATTENTION8H2_SOURCE_EDGE_V1_ARTIFACT.pt", map_location="cpu", weights_only=True)["values"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        assert json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())["pred_a"]
        print("240bodyforwards;native/fullcity/routing/value/additive;CPUcontrol", cpu_control())
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
    a8inputs, factorial_errors, mixed_norms, outside = [], [], [], []
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
        keyd, valued, firstd = xr.clone(), xr.clone(), fr.clone()
        keyd[:, city] = xd[:, city]
        valued[:, city] = xd[:, city]
        firstd[:, city] = fd[:, city]
        r0, v0 = head_factor_parts(attn8, xr, xr, xr, fr)
        rd, vd = head_factor_parts(attn8, xr, keyd, valued, firstd)
        factors = factorial_channels(r0, rd, v0[:, None], vd[:, None])
        factorial_errors.append(float((factors["additive"] + factors["mixed"] - factors["donor"]).norm() / factors["donor"].norm().clamp_min(1e-8)))
        mixed_norms.append(float(select_sources(factors["mixed"], city[None]).norm()))
        recipient = select_sources(factors["recipient"], city[None])
        chosen = {1: "donor", 2: "routing", 3: "value", 4: "additive"}[arm]
        changed = select_sources(factors[chosen], city[None])
        write_delta = F.linear(changed - recipient, weight)
        hybrid = x.clone()
        hybrid[:, framing] += write_delta[:, framing]
        outside.append(float((hybrid[:, ~framing] - x[:, ~framing]).abs().max()))
        mixed = module.lambdas[0] * hybrid + module.lambdas[1] * x0
        context["hybrid_current"] = F.rms_norm(mixed, (mixed.size(-1),))
        return None

    hooks = [
        attn8.register_forward_pre_hook(a8pre),
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
    values = torch.zeros(5, 48, 6, dtype=torch.float64)
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
        for arm in range(1, 5):
            context["arm"] = arm
            for i in range(48):
                context["i"] = i
                forward(i, arm)
    finally:
        for hook in hooks:
            hook.remove()
    assert count == 240 and bool(torch.isfinite(values).all())
    effect = values - values[0]
    anchor = max(float((values[0, :, :2] - prior[0, :, :2]).abs().max()), float((values[1, :, :2] - prior[2, :, :2]).abs().max()))
    cells = []
    for family in range(2):
        ix = [i for i, row in enumerate(rows) if row["family"] == family]
        native = values[0, ix[::2], 0] - values[0, ix[1::2], 0]
        cue = effect[:, ix[::2], 0] - effect[:, ix[1::2], 0]
        full = cue[1]
        target = float(full.square().mean().sqrt())
        rel = lambda arm: float((cue[arm] - full).norm() / full.norm().clamp_min(1e-8))
        routing_error, value_error, additive_error = rel(2), rel(3), rel(4)
        controls = {ARMS[arm]: {READOUTS[j][0]: float(effect[arm, ix, j].square().mean().sqrt() / max(target, 1e-8)) for j in range(2, 6)} for arm in range(1, 5)}
        cells.append({
            "family": family, "native_positive_pairs": int((native > 0).sum()),
            "native_cue_mean": float(native.mean()), "full_city_target_rms": target,
            "routing_to_full_cue_error": routing_error, "value_to_full_cue_error": value_error,
            "additive_no_mixed_to_full_cue_error": additive_error,
            "control_ratios_to_full_target": controls,
            "capability_pass": int((native > 0).sum()) >= 10 and target >= 1e-5,
            "mixed_omissible_pass": additive_error <= .1,
            "routing_hypothesis_pass": routing_error <= .35 and value_error >= .5,
            "value_hypothesis_pass": value_error <= .35 and routing_error >= .5,
        })
    torch.save({"values": values, "effects": effect}, artifact)
    result = {
        "pred_a": anchor <= 1e-5 and max(factorial_errors) <= 1e-5 and min(mixed_norms) >= 1e-8 and max(outside) == 0 and count == 240 and bool(torch.isfinite(values).all()),
        "pred_b": all(c["capability_pass"] for c in cells),
        "pred_c": all(c["mixed_omissible_pass"] for c in cells),
        "pred_d": all(c["routing_hypothesis_pass"] for c in cells),
        "pred_e": all(c["value_hypothesis_pass"] for c in cells),
        "max_frozen_anchor_error": anchor, "max_factorial_error": max(factorial_errors),
        "min_mixed_channel_norm": min(mixed_norms), "max_outside_hybrid_delta": max(outside),
        "families": cells, "body_forwards": count, "seconds": time.perf_counter() - start,
        "artifact_sha256": digest(artifact), "source_shas": binding,
        "arm_order": ARMS, "readout_order": [x[0] for x in READOUTS],
        "scope": "Recipient-query-fixed head8.2 city-source routing/value factorial propagated through framing writes, isolated head9.8-O current values, and native suffix. No whole-head sufficiency, corpus OOD, compression adoption, or quantization.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
