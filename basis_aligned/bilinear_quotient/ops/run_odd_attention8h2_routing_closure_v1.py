#!/usr/bin/env python3
# BQGATE:96bodyforwards;96prefixes;180seconds;no fitting.
"""pred_a routing/write replay; pred_b export integrity; pred_c unfavorable price; pred_d favorable price."""
import json, os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F

from attention8h2_rankone_edge_v1 import direct_write, rankone_write
from attention8h2_routing_closure_v1 import routing
from odd_contextual_positions_v1 import contextual_masks
from regional_cue_row_check_v1 import validate
from run_even_value_factorial_native_v1 import cpu_control
from run_odd_attention8h2_city_key_value_v1 import head_factor_parts
from sparse_path_stability_atlas_v1 import digest

STEM = "ODD_ATTENTION8H2_ROUTING_CLOSURE_V1"


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(k) == value for k, value in binding.items())
    rows = json.loads((P / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ROWS.json").read_text())["rows"]
    assert len(rows) == 96
    validate(rows)
    contexts = contextual_masks(rows)
    parent = torch.load(P / "ODD_ATTENTION8H2_RANKONE_EDGE_V1_PROGRAM.pt", map_location="cpu", weights_only=True)
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        assert json.loads((P / "ODD_ATTENTION8H2_ROUTING_CLOSURE_V2_CPU_CONTROL.json").read_text())["pred_a"]
        print("96bodyforwards;native routing capture/export;CPUcontrol", cpu_control())
        return
    out, program_path, artifact_path = P / (STEM + "_RESULT.json"), P / (STEM + "_PROGRAM.pt"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not program_path.exists() and not artifact_path.exists()
    start = time.perf_counter()
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    attn8 = model.transformer.h[8].attn
    captured = []

    def a8pre(module, args):
        captured.append((args[0].detach().cpu(), args[1].detach().cpu()))

    hook = attn8.register_forward_pre_hook(a8pre)
    count = 0
    try:
        for row in rows:
            ids = torch.tensor([row["ids"]], device="cuda")
            x = F.rms_norm(model.transformer.wte(ids), (1152,))
            x0, v1 = x, None
            for block in model.transformer.h:
                x, v1 = block(x, v1, x0)
            count += 1
    finally:
        hook.remove()
    program = dict(parent)
    program.update({
        "q1_weight": attn8.c_q.weight[2 * 128:3 * 128].detach().cpu(),
        "k1_weight": attn8.c_k.weight[2 * 128:3 * 128].detach().cpu(),
        "q2_weight": attn8.c_q2.weight[2 * 128:3 * 128].detach().cpu(),
        "k2_weight": attn8.c_k2.weight[2 * 128:3 * 128].detach().cpu(),
    })
    gpu_program = {key: value.to("cuda") if isinstance(value, torch.Tensor) else value for key, value in program.items()}
    routing_errors, write_errors = [], []
    representatives = []
    for i, row in enumerate(rows):
        donor = i ^ 1
        city_mask = contexts[i]["city"].to("cuda")
        city = int(torch.nonzero(city_mask)[0])
        xr, fr = [tensor.to("cuda") for tensor in captured[i]]
        _, fd = [tensor.to("cuda") for tensor in captured[donor]]
        native_routing, _ = head_factor_parts(attn8, xr, xr, xr, fr)
        native_routing = native_routing[:, :, city]
        candidate_routing = routing(gpu_program, xr, city)
        routing_errors.append(float((candidate_routing - native_routing).norm() / native_routing.norm().clamp_min(1e-8)))
        first0 = fr.view(1, len(row["ids"]), 9, 128)[:, city, 2]
        firstd = fd.view(1, len(row["ids"]), 9, 128)[:, city, 2]
        delta_value = attn8.lamb * (firstd - first0)
        direct = direct_write(native_routing, delta_value, attn8.c_proj.weight[:, 2 * 128:3 * 128])
        candidate = rankone_write(candidate_routing, delta_value, attn8.c_proj.weight[:, 2 * 128:3 * 128])
        write_errors.append(float((candidate - direct).norm() / direct.norm().clamp_min(1e-8)))
        if row["endpoint"] == 0 and row["pair"] == 0 and row["cue"] == "British":
            ids0 = torch.tensor([row["ids"]], device="cuda")
            idsd = torch.tensor([rows[donor]["ids"]], device="cuda")
            representatives.append({
                "family": row["family"], "current": xr.cpu(), "city_index": city,
                "recipient_embedding": F.rms_norm(model.transformer.wte(ids0), (1152,))[:, city].cpu(),
                "donor_embedding": F.rms_norm(model.transformer.wte(idsd), (1152,))[:, city].cpu(),
                "direct_write": direct.cpu(),
            })
    lengths = sorted(set(len(row["ids"]) for row in rows))
    prices = []
    for tokens in lengths:
        dense = tokens * 1152
        closed = tokens * 1152 + 2 * 1152 + 1
        prices.append({"tokens": tokens, "dense_write_scalars": dense, "closed_runtime_input_scalars": closed, "closed_to_dense_ratio": closed / dense})
    maps_match = all(torch.equal(program[key], parent[key]) for key in ("value_weight", "output_weight", "mixture"))
    finite = all(torch.isfinite(value).all() for value in program.values() if isinstance(value, torch.Tensor))
    torch.save(program, program_path)
    torch.save({"representatives": representatives}, artifact_path)
    result = {
        "pred_a": max(routing_errors) <= 1e-5 and max(write_errors) <= 1e-5,
        "pred_b": maps_match and finite and count == 96,
        "pred_c": all(item["closed_to_dense_ratio"] >= 1 for item in prices),
        "pred_d": all(item["closed_to_dense_ratio"] <= .5 for item in prices),
        "max_routing_relative_error": max(routing_errors), "max_write_relative_error": max(write_errors),
        "parent_maps_bitwise_equal": maps_match, "prices": prices,
        "static_weight_scalars": 6 * 128 * 1152 + 1, "body_forwards": count,
        "seconds": time.perf_counter() - start, "program_sha256": digest(program_path),
        "artifact_sha256": digest(artifact_path), "source_shas": binding,
        "scope": "Exact conditional closure of head8.2 city routing and inherited rank-one write from normalized block8 current state and normalized city embeddings. Upstream state generation, downstream O and suffix remain external.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
