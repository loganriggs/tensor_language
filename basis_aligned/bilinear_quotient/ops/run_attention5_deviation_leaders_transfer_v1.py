#!/usr/bin/env python3
# BQGATE:49documentforwards;2frozencorpora;180seconds;no refit or head selection.
"""Prospective natural/code transfer of frozen attention5 deviation leaders."""
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

STEM = "ATTENTION5_DEVIATION_LEADERS_TRANSFER_V1"
D, V, T, NH, HD, CHUNK = 1152, 50304, 257, 9, 128, 8
LEADERS = (6, 7, 3)


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
    parent = json.loads((P / "ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_RESULT.json").read_text())
    saved = torch.load(P / "ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_ARTIFACT.pt", map_location="cpu", weights_only=True)
    assert control["pred_a"] and tuple(control["leaders"]) == LEADERS and parent["singleton_order"][:3] == ["6", "7", "3"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print("49documentforwards;final-natural+ood-code;frozen leaders 6,7,3;no refit")
        return
    out, artifact = P / (STEM + "_RESULT.json"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not artifact.exists()
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval()
    mean_z = saved["mean_head_input"].cuda().float()
    context = {"mode": "native", "heads": ()}
    cproj = model.transformer.h[5].attn.c_proj

    def hook(module, args, output):
        if context["mode"] == "native":
            return output
        z = args[0].view(args[0].shape[0], args[0].shape[1], NH, HD)
        weight = module.weight.float()
        bias = None if module.bias is None else module.bias.float()
        changed = F.linear(mean_z.flatten(), weight, bias).view(1, 1, D).expand(z.shape[0], z.shape[1], D).clone()
        for h in context["heads"]:
            changed += F.linear(z[:, :, h].float() - mean_z[h], weight[:, h * HD:(h + 1) * HD], None)
        return changed.to(output.dtype)

    handle = cproj.register_forward_hook(hook)
    forwards = 0
    def losses(docs, mode, heads=()):
        nonlocal forwards
        context["mode"], context["heads"] = mode, tuple(heads)
        pieces = []
        for start in range(0, len(docs), CHUNK):
            batch = docs[start:start + CHUNK]
            idx, target = batch[:, :T - 1].cuda(), batch[:, 1:T].cuda()
            logits = doc_forward(model, idx); forwards += 1
            loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none")
            pieces.append(loss.view(batch.shape[0], -1).mean(1).cpu())
        return torch.cat(pieces).double()

    start_time = time.perf_counter()
    corpora, saved_losses = {}, {}
    complement = tuple(h for h in range(NH) if h not in LEADERS)
    for label, filename in (("final_natural", "final_natural.pt"), ("ood_code", "ood_code.pt")):
        obj = torch.load(BQ / ".rowcache_terminal_copy_induction_v2" / filename, map_location="cpu", weights_only=False)
        docs = (obj["rows"] if isinstance(obj, dict) else obj)[:24].long()
        native = losses(docs, "native")
        mean = losses(docs, "mean")
        leaders = losses(docs, "subset", LEADERS)
        rest = losses(docs, "subset", complement)
        all_heads = losses(docs, "all", range(NH))
        singles = {str(h): losses(docs, "subset", (h,)) for h in LEADERS}
        def recovery(arm, sl=slice(None)):
            denom = float((mean[sl] - native[sl]).mean())
            return float((denom - float((arm[sl] - native[sl]).mean())) / denom)
        corpora[label] = {
            "mean_damage": float((mean - native).mean()),
            "leaders_recovery": recovery(leaders), "complement_recovery": recovery(rest),
            "all_restore_ce_error": float((all_heads - native).mean()),
            "singleton_recovery": {h: recovery(x) for h, x in singles.items()},
            "half_leaders_gt_complement": [recovery(leaders, sl) > recovery(rest, sl) for sl in (slice(0, 12), slice(12, 24))],
            "mean_ce": {"native": float(native.mean()), "mean": float(mean.mean()), "leaders": float(leaders.mean()), "complement": float(rest.mean()), "all": float(all_heads.mean())},
        }
        saved_losses[label] = {"native": native, "mean": mean, "leaders": leaders, "complement": rest, "all": all_heads, "singletons": singles}
    context["mode"] = "native"
    sample_obj = torch.load(BQ / ".rowcache_terminal_copy_induction_v2/final_natural.pt", map_location="cpu", weights_only=False)
    sample = sample_obj["rows"][:CHUNK]
    idx, target = sample[:, :T - 1].cuda(), sample[:, 1:T].cuda()
    manual = float(F.cross_entropy(doc_forward(model, idx).reshape(-1, V), target.reshape(-1))); forwards += 1
    module = float(model(idx.contiguous(), target.contiguous()))
    handle.remove()
    result = {
        "pred_a": abs(manual - module) <= .01 and all(abs(x["all_restore_ce_error"]) <= .005 for x in corpora.values()),
        "pred_b": all(x["mean_damage"] >= .05 for x in corpora.values()),
        "pred_c": all(x["leaders_recovery"] >= .50 for x in corpora.values()),
        "pred_d": all(x["complement_recovery"] <= .50 for x in corpora.values()),
        "pred_e": all(all(v > 0 for v in x["singleton_recovery"].values()) and all(x["half_leaders_gt_complement"]) for x in corpora.values()),
        "manual_module_ce_error": abs(manual - module), "corpora": corpora,
        "body_forwards": forwards, "seconds": time.perf_counter() - start_time,
        "source_shas": binding,
        "scope": "Prospective transfer of discovery-selected attention5 mean-deviation heads6/7/3 to untouched natural and code caches; no refit or factor localization.",
    }
    torch.save(saved_losses, artifact); result["artifact_sha256"] = digest(artifact)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
