#!/usr/bin/env python3
# BQGATE:49documentforwards;24fit+24disjointeval;180seconds;noheadselection.
"""Identify which native heads carry attention5's input-dependent mean deviation."""
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(BQ), str(ROOT)]
import torch
import torch.nn.functional as F

from sparse_path_stability_atlas_v1 import digest

STEM = "ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V1"
D, V, T, NH, HD = 1152, 50257, 256, 9, 128
FIT_DOCS, EVAL_DOCS, CHUNK = 24, 24, 8
FOCAL = (5, 7)


@torch.no_grad()
def doc_forward(model, idx):
    x = F.rms_norm(model.transformer.wte(idx), (D,))
    x0, first = x, None
    for block in model.transformer.h:
        x, first = block(x, first, x0)
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(path) == value for path, value in binding.items())
    control = json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())
    old_direction = json.loads((BQ / "circuit_battery_attn5_direction_identity_results.json").read_text())
    old_heads = json.loads((BQ / "circuit_battery_attn5_heldout_surrogate_results.json").read_text())
    assert control["pred_a"] and old_direction["summary"]["const_damage"] >= .10
    assert old_heads["keep_heads"] == [5, 7]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print("49documentforwards;24fit+24disjointeval;arms=native,mean,h57,other,all,9singletons")
        return
    out = P / (STEM + "_RESULT.json")
    artifact = P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not artifact.exists()
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    docs_obj = torch.load(BQ / ".rowcache_terminal_copy_induction_v2/fit_natural.pt", map_location="cpu", weights_only=False)
    docs = (docs_obj["rows"] if isinstance(docs_obj, dict) else docs_obj).long()
    fit, evaluate = docs[:FIT_DOCS], docs[FIT_DOCS:FIT_DOCS + EVAL_DOCS]
    assert fit.shape[0] == FIT_DOCS and evaluate.shape[0] == EVAL_DOCS
    context = {"mode": "native", "heads": (), "sum_z": torch.zeros(NH, HD, device="cuda"), "count": 0, "max_rebuild": 0.0}
    mean_z = None
    cproj = model.transformer.h[5].attn.c_proj

    def hook(module, args, output):
        nonlocal mean_z
        z = args[0].view(args[0].shape[0], args[0].shape[1], NH, HD)
        if context["mode"] == "collect":
            context["sum_z"] += z.float().sum((0, 1))
            context["count"] += z.shape[0] * z.shape[1]
            return output
        if context["mode"] == "native":
            return output
        assert mean_z is not None
        weight = module.weight.float()
        bias = None if module.bias is None else module.bias.float()
        base = F.linear(mean_z.flatten(), weight, bias).view(1, 1, D)
        changed = base.expand(z.shape[0], z.shape[1], D).clone()
        for h in context["heads"]:
            changed += F.linear(z[:, :, h].float() - mean_z[h], weight[:, h * HD:(h + 1) * HD], None)
        if context["mode"] == "all":
            context["max_rebuild"] = max(context["max_rebuild"], float((changed - output.float()).abs().max()))
        return changed.to(output.dtype)

    handle = cproj.register_forward_hook(hook)
    forwards = 0

    def losses(docset, mode, heads=()):
        nonlocal forwards
        context["mode"], context["heads"] = mode, tuple(heads)
        pieces = []
        for start in range(0, len(docset), CHUNK):
            batch = docset[start:start + CHUNK]
            idx, target = batch[:, :T - 1].cuda(), batch[:, 1:T].cuda()
            logits = doc_forward(model, idx)
            forwards += 1
            row = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none")
            pieces.append(row.view(batch.shape[0], -1).mean(1).cpu())
        return torch.cat(pieces).double()

    start_time = time.perf_counter()
    losses(fit, "collect")
    mean_z = context["sum_z"] / context["count"]
    native = losses(evaluate, "native")
    mean = losses(evaluate, "mean")
    h57 = losses(evaluate, "subset", FOCAL)
    other_heads = tuple(h for h in range(NH) if h not in FOCAL)
    other = losses(evaluate, "subset", other_heads)
    all_heads = losses(evaluate, "all", range(NH))
    singles = {str(h): losses(evaluate, "subset", (h,)) for h in range(NH)}

    # Matched model-loss instrument on the first evaluation chunk.
    context["mode"] = "native"
    idx0, target0 = evaluate[:CHUNK, :T - 1].cuda(), evaluate[:CHUNK, 1:T].cuda()
    manual = float(F.cross_entropy(doc_forward(model, idx0).reshape(-1, V), target0.reshape(-1)))
    forwards += 1
    module = float(model(idx0.contiguous(), target0.contiguous()))
    handle.remove()

    mean_damage = float((mean - native).mean())
    def recovery(arm, sl=slice(None)):
        denom = float((mean[sl] - native[sl]).mean())
        return float((denom - float((arm[sl] - native[sl]).mean())) / denom)
    rec_h57, rec_other = recovery(h57), recovery(other)
    singleton_recovery = {h: recovery(v) for h, v in singles.items()}
    order = sorted(singleton_recovery, key=singleton_recovery.get, reverse=True)
    half_ordering = []
    for sl in (slice(0, 12), slice(12, 24)):
        half_ordering.append(recovery(h57, sl) > recovery(other, sl))
    pred_a = abs(manual - module) <= .01 and context["max_rebuild"] <= 1e-3 and abs(float((all_heads - native).mean())) <= .005
    result = {
        "pred_a": pred_a,
        "pred_b": mean_damage >= .08,
        "pred_c": rec_h57 >= .50,
        "pred_d": rec_other <= .50,
        "pred_e": {"5", "7"}.issubset(set(order[:3])) and all(half_ordering),
        "manual_module_ce_error": abs(manual - module),
        "max_head_sum_rebuild_error": context["max_rebuild"],
        "all_restore_ce_error": float((all_heads - native).mean()),
        "mean_only_damage": mean_damage,
        "h57_recovery": rec_h57,
        "other7_recovery": rec_other,
        "half_h57_gt_other7": half_ordering,
        "singleton_recovery": singleton_recovery,
        "singleton_order": order,
        "mean_ce": {"native": float(native.mean()), "mean": float(mean.mean()), "h57": float(h57.mean()), "other7": float(other.mean()), "all": float(all_heads.mean())},
        "body_forwards": forwards,
        "seconds": time.perf_counter() - start_time,
        "source_shas": binding,
        "scope": "Disjoint natural-document head decomposition of attention5 mean deviations; no rank fit, task semantics, factor localization, or installed surrogate.",
    }
    torch.save({"mean_head_input": mean_z.cpu(), "mean_loss": mean, "native_loss": native, "h57_loss": h57, "other7_loss": other, "singleton_losses": singles}, artifact)
    result["artifact_sha256"] = digest(artifact)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
