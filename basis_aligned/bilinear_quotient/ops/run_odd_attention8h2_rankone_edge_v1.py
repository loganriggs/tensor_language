#!/usr/bin/env python3
# BQGATE:96bodyforwards;96prefixes;180seconds;no fitting.
"""pred_a native factor/reconstruction; pred_b bound parents; pred_c price."""
import json, os, signal, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(ROOT)]
import torch
import torch.nn.functional as F

from attention8h2_rankone_edge_v1 import direct_write, rankone_write, price
from odd_contextual_positions_v1 import contextual_masks
from regional_cue_row_check_v1 import validate
from run_even_value_factorial_native_v1 import cpu_control
from run_odd_attention8h2_city_key_value_v1 import head_factor_parts
from sparse_path_stability_atlas_v1 import digest

STEM = "ODD_ATTENTION8H2_RANKONE_EDGE_V1"
PARENTS = [
    "ODD_ATTENTION8H2_CITY_VALUE_SOURCE_V1_RESULT.json",
    "ODD_ATTENTION8H2_CHAIN_FRESH_V1_RESULT.json",
    "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_RESULT.json",
]
EXPECTED_PARENT_VECTORS = {
    PARENTS[0]: [True, True, False, True, True],
    PARENTS[1]: [True, True, True, True, True, True],
    PARENTS[2]: [True, True, False, False, True, True],
}


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(k) == v for k, v in binding.items())
    rows = json.loads((P / "ODD_ATTENTION8H2_DESTINATION_ROLE_V1_ROWS.json").read_text())["rows"]
    assert len(rows) == 96
    validate(rows)
    contexts = contextual_masks(rows)
    parent_verdicts = {name: {key: value for key, value in json.loads((P / name).read_text()).items() if key.startswith("pred_")} for name in PARENTS}
    parent_vectors = {name: [value for key, value in sorted(verdict.items())] for name, verdict in parent_verdicts.items()}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        assert json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())["pred_a"]
        print("96bodyforwards;native capture/factor/export;CPUcontrol", cpu_control())
        return
    out = P / (STEM + "_RESULT.json")
    program_path = P / (STEM + "_PROGRAM.pt")
    artifact_path = P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not program_path.exists() and not artifact_path.exists()
    start = time.perf_counter()
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    attn0, attn8 = model.transformer.h[0].attn, model.transformer.h[8].attn
    captured, preproj = [], []

    def a8pre(module, args):
        captured.append((args[0].detach().cpu(), args[1].detach().cpu()))

    def cpre(module, args):
        preproj.append(args[0].detach().cpu())

    hooks = [attn8.register_forward_pre_hook(a8pre), attn8.c_proj.register_forward_pre_hook(cpre)]
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
        for hook in hooks:
            hook.remove()
    assert count == 96 and len(captured) == len(preproj) == 96
    output_weight = attn8.c_proj.weight[:, 2 * 128:3 * 128]
    value_weight = attn0.c_v.weight[2 * 128:3 * 128]
    factor_errors, inherited_errors, source_replays = [], [], []
    representatives = []
    for i, row in enumerate(rows):
        donor = i ^ 1
        city = contexts[i]["city"].to("cuda")
        xr, fr = [x.to("cuda") for x in captured[i]]
        _, fd = [x.to("cuda") for x in captured[donor]]
        routing, mixed_value = head_factor_parts(attn8, xr, xr, xr, fr)
        channels = routing[..., None] * mixed_value[:, None]
        source_replays.append(float((channels.sum(-2) - preproj[i].to("cuda")[..., 2 * 128:3 * 128]).norm() / channels.sum(-2).norm().clamp_min(1e-8)))
        first0 = fr.view(1, len(row["ids"]), 9, 128)[:, :, 2]
        firstd = fd.view(1, len(row["ids"]), 9, 128)[:, :, 2]
        delta_value = attn8.lamb * (firstd[:, city] - first0[:, city]).squeeze(1)
        city_index = int(torch.nonzero(city)[0])
        city_routing = routing[:, :, city_index]
        direct = direct_write(city_routing, delta_value, output_weight)
        factored = rankone_write(city_routing, delta_value, output_weight)
        factor_errors.append(float((direct - factored).norm() / direct.norm().clamp_min(1e-8)))
        ids0 = torch.tensor([row["ids"]], device="cuda")
        idsd = torch.tensor([rows[donor]["ids"]], device="cuda")
        embed0 = F.rms_norm(model.transformer.wte(ids0), (1152,))[:, city_index]
        embedd = F.rms_norm(model.transformer.wte(idsd), (1152,))[:, city_index]
        reconstructed = attn8.lamb * F.linear(embedd - embed0, value_weight)
        inherited_errors.append(float((reconstructed - delta_value).norm() / delta_value.norm().clamp_min(1e-8)))
        if row["endpoint"] == 0 and row["pair"] == 0 and row["cue"] == "British":
            representatives.append({"family": row["family"], "routing": city_routing.cpu(), "recipient_embedding": embed0.cpu(), "donor_embedding": embedd.cpu(), "direct_write": direct.cpu()})
    lengths = sorted(set(len(row["ids"]) for row in rows))
    prices = [price(tokens) for tokens in lengths]
    program = {
        "value_weight": value_weight.detach().cpu(), "output_weight": output_weight.detach().cpu(),
        "mixture": attn8.lamb.detach().cpu(), "head": 2, "source_layer": 0,
        "writer_layer": 8, "value_width": 128, "residual_width": 1152,
    }
    torch.save(program, program_path)
    torch.save({"representatives": representatives}, artifact_path)
    finite = all(torch.isfinite(tensor).all() for tensor in program.values() if isinstance(tensor, torch.Tensor))
    result = {
        "pred_a": max(source_replays) <= 1e-5 and max(factor_errors) <= 1e-5 and max(inherited_errors) <= 1e-5 and finite and count == 96,
        "pred_b": parent_vectors == EXPECTED_PARENT_VECTORS,
        "pred_c": all(item["interface_fraction_saved"] >= .9 and item["multiply_fraction_saved"] >= .9 for item in prices),
        "max_head_source_replay_error": max(source_replays), "max_rankone_relative_error": max(factor_errors),
        "max_inherited_reconstruction_error": max(inherited_errors), "parent_verdicts": parent_verdicts,
        "prices": prices, "static_weight_scalars": 2 * 1152 * 128 + 1,
        "body_forwards": count, "seconds": time.perf_counter() - start,
        "program_sha256": digest(program_path), "artifact_sha256": digest(artifact_path),
        "source_shas": binding,
        "scope": "Conditional exact extraction of the inherited-city head8.2 rank-one write. Caller supplies normalized embeddings and native destination routing; downstream head9.8-O and suffix remain external.",
    }
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "source_shas"}), flush=True)
    signal.alarm(0)


if __name__ == "__main__":
    main()
