#!/usr/bin/env python3
# BQGATE:73documentforwards;natural discovery/code heldout;180seconds;fixed rank3.
"""Operational final-vocabulary response quotient for all attention5 deviations."""
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P, BQ = ROOT / "basis_aligned/polynomial_causal", ROOT / "basis_aligned/bilinear_quotient"
sys.path[:0] = [str(Path(__file__).parent), str(P), str(BQ), str(ROOT)]
import torch
import torch.nn.functional as F
from sparse_path_stability_atlas_v1 import digest

STEM = "ATTENTION5_DEVIATION_RESPONSE_QUOTIENT_V1"
D, V, T, NH, HD, CHUNK = 1152, 50304, 257, 9, 128, 8


@torch.no_grad()
def doc_forward(model, idx):
    x = F.rms_norm(model.transformer.wte(idx), (D,)); x0, first = x, None
    for block in model.transformer.h:
        x, first = block(x, first, x0)
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (D,))) / 30.0)


def rel(x, y):
    return float((x - y).norm() / y.norm().clamp_min(1e-12))


@torch.no_grad()
def main():
    binding = json.loads((P / (STEM + "_BINDING.json")).read_text())["files"]
    assert all(digest(path) == value for path, value in binding.items())
    control = json.loads((P / (STEM + "_CPU_CONTROL.json")).read_text())
    saved = torch.load(P / "ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_ARTIFACT.pt", map_location="cpu", weights_only=True)
    assert control["pred_a"]
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print("73documentforwards;9singleton+mean+all+native;natural discovery;code heldout;rank3 frozen")
        return
    out, artifact = P / (STEM + "_RESULT.json"), P / (STEM + "_ARTIFACT.pt")
    assert not out.exists() and not artifact.exists()
    signal.alarm(180); torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast
    model = load_model_fast().cuda().eval(); mean_z = saved["mean_head_input"].cuda().float()
    context = {"mode": "native", "heads": ()}; cproj = model.transformer.h[5].attn.c_proj

    def hook(module, args, output):
        if context["mode"] == "native": return output
        z = args[0].view(args[0].shape[0], args[0].shape[1], NH, HD); weight = module.weight.float()
        bias = None if module.bias is None else module.bias.float()
        changed = F.linear(mean_z.flatten(), weight, bias).view(1, 1, D).expand(z.shape[0], z.shape[1], D).clone()
        for h in context["heads"]:
            changed += F.linear(z[:, :, h].float() - mean_z[h], weight[:, h * HD:(h + 1) * HD], None)
        return changed.to(output.dtype)

    handle = cproj.register_forward_hook(hook); forwards = 0
    def evaluate(docs, mode, heads=()):
        nonlocal forwards
        context["mode"], context["heads"] = mode, tuple(heads); losses, total = [], torch.zeros(V, dtype=torch.float64); count = 0
        for start in range(0, len(docs), CHUNK):
            batch = docs[start:start + CHUNK]; idx, target = batch[:, :T - 1].cuda(), batch[:, 1:T].cuda()
            logits = doc_forward(model, idx); forwards += 1
            loss = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none")
            losses.append(loss.view(batch.shape[0], -1).mean(1).cpu())
            total += logits.double().sum((0, 1)).cpu(); count += logits.shape[0] * logits.shape[1]
        return torch.cat(losses).double(), total / count

    start_time = time.perf_counter(); banks, metrics = {}, {}
    for label, filename in (("final_natural", "final_natural.pt"), ("ood_code", "ood_code.pt")):
        obj = torch.load(BQ / ".rowcache_terminal_copy_induction_v2" / filename, map_location="cpu", weights_only=False)
        docs = obj["rows"][:24].long(); native_loss, native_logits = evaluate(docs, "native")
        mean_loss, mean_logits = evaluate(docs, "mean"); all_loss, all_logits = evaluate(docs, "all", range(NH))
        single = {str(h): evaluate(docs, "subset", (h,)) for h in range(NH)}
        response = torch.stack([single[str(h)][1] - mean_logits for h in range(NH)])
        denom = float((mean_loss - native_loss).mean())
        recoveries = {str(h): float((denom - float((single[str(h)][0] - native_loss).mean())) / denom) for h in range(NH)}
        full = all_logits - mean_logits
        metrics[label] = {"mean_damage": denom, "all_restore_ce_error": float((all_loss - native_loss).mean()),
                          "composition_error": rel(response.sum(0), full), "positive_recovery_heads": sum(x > 0 for x in recoveries.values()),
                          "singleton_recovery": recoveries}
        banks[label] = {"response": response, "full": full}
    u, s, vh = torch.linalg.svd(banks["final_natural"]["response"], full_matrices=False)
    rank3 = vh[:3]; discovery_energy = float((s[:3] ** 2).sum() / (s ** 2).sum())
    code = banks["ood_code"]["response"]; code_reconstruction = rel((code @ rank3.T) @ rank3, code)
    cross_cosines = []
    for h in range(NH):
        a, b = banks["final_natural"]["response"][h], code[h]
        cross_cosines.append(float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12)))
    median_cosine = float(torch.tensor(cross_cosines).median())
    context["mode"] = "native"
    sample = torch.load(BQ / ".rowcache_terminal_copy_induction_v2/final_natural.pt", map_location="cpu", weights_only=False)["rows"][:CHUNK]
    idx, target = sample[:, :T - 1].cuda(), sample[:, 1:T].cuda()
    manual = float(F.cross_entropy(doc_forward(model, idx).reshape(-1, V), target.reshape(-1))); forwards += 1
    module = float(model(idx.contiguous(), target.contiguous())); handle.remove()
    result = {"pred_a": abs(manual-module) <= .01 and all(abs(x["all_restore_ce_error"]) <= .005 for x in metrics.values()),
              "pred_b": discovery_energy >= .80, "pred_c": code_reconstruction <= .50,
              "pred_d": all(x["composition_error"] <= .25 for x in metrics.values()),
              "pred_e": median_cosine >= .50 and all(x["positive_recovery_heads"] >= 7 for x in metrics.values()),
              "manual_module_ce_error": abs(manual-module), "discovery_rank3_energy": discovery_energy,
              "discovery_singular_values": s.tolist(), "code_rank3_reconstruction_error": code_reconstruction,
              "cross_corpus_head_cosines": cross_cosines, "median_cross_corpus_head_cosine": median_cosine,
              "corpora": metrics, "body_forwards": forwards, "seconds": time.perf_counter()-start_time,
              "source_shas": binding, "scope": "Aggregate final-vocabulary operational quotient for nine attention5 mean-deviation interventions; natural discovery/code heldout; no semantic or source claim."}
    torch.save({"banks": banks, "rank3": rank3}, artifact); result["artifact_sha256"] = digest(artifact)
    out.write_text(json.dumps(result, indent=2) + "\n"); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
