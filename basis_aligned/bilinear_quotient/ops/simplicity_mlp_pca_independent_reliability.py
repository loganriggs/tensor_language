"""RUNG453 -- INDEPENDENT RELIABILITY OF COMPLETE MLP-PCA CONSEQUENCES."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from receipt import dump
from simplicity_mlp0_complete_candidate_consequences import factored, metrics
from simplicity_mlp_pca_complete_candidate_consequences import build_variants


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
OUT = ROOT / "simplicity_mlp_pca_independent_reliability_results.json"
BUNDLE = ROOT / "simplicity_mlp_pca_independent_reliability_bundle.pt"
ROWS = ROOT / ".rowcache_simplicity_mlp_pca_reliability_v1/reliability_192.pt"
ROWS_RECEIPT = ROOT / "simplicity_mlp_pca_reliability_v1_rows_receipt.json"
SPEC = ROOT / "simplicity_mlp_pca_reliability_spec_v1.json"
OLD_RESULT = ROOT / "simplicity_mlp_pca_complete_candidate_consequences_results.json"
BANK = POLY / "prospective_consequence_candidate_bank_v1.json"
GENERIC_FIT = ROOT / ".rowcache/fineweb_n480_skip80.pt"
MEAN_FIT = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
CENSUS = ROOT / "census_state_diverse.pt"
FACTOR = ROOT / "mlp16_rank2_quadratic_factored.pt"
CONDITIONS = ("unablated", "knockout", "partner")
CONDITION_FILES = {name: ROOT / f"simplicity_mlp_pca_reliability_{name}.pt" for name in CONDITIONS}
IDS = (
    "mlp_pca_p0_8_r256", "mlp_pca_p0_17_r256", "mlp_pca_p8_17_r256",
    "mlp_pca_p8_17_r384", "mlp_pca_p8_17_r512", "mlp_pca_grad32", "mlp_pca_grad64",
)
PRICES = {
    "mlp_pca_p0_8_r256": 531_927_350, "mlp_pca_p0_17_r256": 531_927_350,
    "mlp_pca_p8_17_r256": 531_927_350, "mlp_pca_p8_17_r384": 533_401_910,
    "mlp_pca_p8_17_r512": 534_876_470, "mlp_pca_grad32": 531_927_350,
    "mlp_pca_grad64": 531_927_350,
}
D = 1152
DOCUMENTS = 192
TOKENS = 256
HASHES = {
    BANK: "e35d5c0aa1dae34173b93ae4d81cafa8317539adfaf7c74bfe7decb068ac47be",
    ROWS_RECEIPT: "d991469476d9f2623f011cfdb57a768fe41bb9589e24df67ccc095497c21ebc3",
    ROWS: "c340a75c27afa0c37f4404c8ddc865275ed626542260c7f84ae14dba7c1ccbbc",
    SPEC: "90ff82093151b1bc71c6f814cf800a6ff167010b9bfbdb33791ebf9b77bdeb5c",
    OLD_RESULT: "0f99365bdb9a21fb4674cc5695d89435127bd3225de7767e4d4d177a6191344e",
    GENERIC_FIT: "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496",
    MEAN_FIT: "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
    CENSUS: "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
    FACTOR: "b9870f738b528e988ff9a1e04cdc6e1096de8ab0dc5fa86bb76229812d9ffb6e",
    ROOT / "ops/cevdump_ct96.py": "1fc1d2a405b94228885921b6085294a7ada609badc6c4c834c92f447d483c932",
    ROOT / "ops/simplicity_mlp_pca_complete_candidate_consequences.py":
        "ed3a120c464b087d607c8fe50c24f7f909eb09ed3032cd0f07c146b989499a82",
    ROOT / "ops/mlp_activation_pca_four_layer_composition.py":
        "d4ae6d01c5a3afd7cfc938a8d2c5ebebb5af9ebb4dfff6da592db288be257213",
    ROOT / "ops/mixed104_pca_fixed_pair_frontier.py":
        "82af814e73d7321b981ba520e7df75d4f8452213c1c75af70a15c809a50aa6b2",
    ROOT / "ops/mixed104_pca_fixed_pair_rank_frontier.py":
        "df52ecc8b7183a9e1a55524a987e806fc6b616d511df05c6ee6da91c7d25bbbd",
    ROOT / "ops/mixed104_pca_certificate_gradient_hybrid.py":
        "9b7959e1be07545de99d51d13d279b18fd470a4f385d8a127f17381920eb9c46",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def pearson(left: list[float], right: list[float]) -> float:
    a = torch.tensor(left, dtype=torch.float64); b = torch.tensor(right, dtype=torch.float64)
    a -= a.mean(); b -= b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def pair_accuracy(values: dict[str, float], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for pair in pairs:
        difference = values[pair["left"]] - values[pair["right"]]
        observed_sign = 1 if difference > 0 else (-1 if difference < 0 else 0)
        passed = observed_sign == pair["expected_sign_left_minus_right"]
        rows.append({"left": pair["left"], "right": pair["right"],
                     "expected_sign_left_minus_right": pair["expected_sign_left_minus_right"],
                     "observed_difference": difference, "replicated": passed})
    count = sum(row["replicated"] for row in rows)
    return {"correct": count, "total": len(rows), "accuracy": count / len(rows), "pairs": rows}


@torch.no_grad()
def child(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(condition)
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import cevdump_ct96 as C
    from bilin18_observed_model_facade import validate_production_model
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    C.m.eval()
    for parameter in C.m.parameters():
        parameter.requires_grad_(False)
    validate_production_model(C.m)
    rows = torch.load(ROWS, map_location="cpu", weights_only=True).long()
    if tuple(rows.shape) != (DOCUMENTS, TOKENS + 1):
        raise RuntimeError("independent reliability row geometry changed")
    variants, diagnostics = build_variants(C, _manual_logits)

    fit = torch.load(MEAN_FIT, map_location="cpu", weights_only=True)
    fit = fit["rows"] if isinstance(fit, dict) else fit
    calibration = fit[:128, :257].long().contiguous()
    mean_sum = torch.zeros(D, device="cuda", dtype=torch.float64); mean_n = 0
    def capture_mean(_module, _args, output):
        nonlocal mean_n
        value = output[0].detach().double().reshape(-1, D)
        mean_sum.add_(value.sum(0)); mean_n += value.shape[0]
    handle = C.m.transformer.h[16].attn.register_forward_hook(capture_mean)
    try:
        for start in range(0, len(calibration), 4):
            _manual_logits(C.m, calibration[start:start + 4, :-1].cuda())
    finally:
        handle.remove()
    mean_value = (mean_sum / mean_n).float()
    factor_cpu = torch.load(FACTOR, map_location="cpu", weights_only=True)
    factor = {key: value.cuda().float() for key, value in factor_cpu.items()}

    native_aux: dict[str, torch.Tensor] = {}
    native_counts = {"knockout": 0, "partner": 0}
    def score_native(mode: str) -> torch.Tensor:
        handles = []
        if mode == "knockout":
            def ko_hook(_module, _args, output):
                native_counts["knockout"] += 1
                return mean_value.expand_as(output[0]).to(output[0].dtype), output[1]
            handles.append(C.m.transformer.h[16].attn.register_forward_hook(ko_hook))
        if mode == "partner":
            def q_hook(_module, args, output):
                native_counts["partner"] += 1
                return factored(args[0], factor).to(output.dtype)
            handles.append(C.m.transformer.h[16].mlp.register_forward_hook(q_hook))
        values = []
        try:
            for start in range(0, len(rows), 4):
                batch = rows[start:start + 4].cuda()
                logits = _manual_logits(C.m, batch[:, :-1])
                values.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                              batch[:, 1:].reshape(-1), reduction="none").cpu())
        finally:
            for item in handles:
                item.remove()
        return torch.cat(values).float()
    if condition == "unablated":
        native_aux["native_ce"] = score_native("native")
        native_aux["native_replay_ce"] = score_native("native")
        native_aux["native_ko_ce"] = score_native("knockout")
        native_aux["partner_ce"] = score_native("partner")

    intervention = {"calls": 0}
    def knockout_hook(_module, _inputs, output):
        if not C.SEL.get("abl_on"):
            return None
        intervention["calls"] += 1
        return mean_value.expand_as(output[0]).to(output[0].dtype), output[1]
    def dummy_hook(_module, _inputs, _output):
        return None
    def partner_hook(_module, args, output):
        if not C.SEL.get("abl_on"):
            return None
        intervention["calls"] += 1
        return factored(args[0], factor).to(output.dtype)

    C.CROWS, C.CBASE, C.NFLAT = rows, torch.zeros(DOCUMENTS * TOKENS), DOCUMENTS * TOKENS
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_projectors": variants[IDS[0]],
        "final_mlp_projector_variants": variants,
        "final_mlp_primary_variant": IDS[0],
    })
    partner_handle = None
    if condition == "knockout":
        C.SEL.update({"ablate_on_census": True, "_ablh": knockout_hook})
    elif condition == "partner":
        C.SEL.update({"ablate_on_census": True, "_ablh": dummy_hook})
        partner_handle = C.m.transformer.h[16].mlp.register_forward_hook(partner_hook)
    try:
        run = C.main()
    finally:
        if partner_handle is not None:
            partner_handle.remove()

    cevs = C.SEL.get("_final_mlp_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_variant_observed", {})
    expected_maps = {
        "mlp_pca_p0_8_r256": {0: 256, 8: 256}, "mlp_pca_p0_17_r256": {0: 256, 17: 256},
        "mlp_pca_p8_17_r256": {8: 256, 17: 256}, "mlp_pca_p8_17_r384": {8: 384, 17: 384},
        "mlp_pca_p8_17_r512": {8: 512, 17: 512}, "mlp_pca_grad32": {8: 256, 17: 256},
        "mlp_pca_grad64": {8: 256, 17: 256},
    }
    wanted = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(item[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for item in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    identity = bool(
        set(cevs) == set(IDS) and set(observed) == set(IDS)
        and all({int(key): int(value) for key, value in observed[name].items()} == expected_maps[name]
                for name in IDS)
        and set(index_sets) == set(range(2, 18)) and all(value == wanted for value in index_sets.values())
        and widths == {104} and not any(name in active for name in ("a0", "a1v", "tailE")))
    if not identity:
        raise RuntimeError("independent complete MLP-PCA candidate identity changed")
    if condition in ("knockout", "partner") and intervention["calls"] != len(IDS) * 48:
        raise RuntimeError(f"{condition} call count changed: {intervention['calls']}")
    payload = {
        "schema": "simplicity_mlp_pca_reliability_condition_v1", "condition": condition,
        "candidate_ce": {name: cevs[name].float().cpu() for name in IDS},
        "native_aux": native_aux, "native_counts": native_counts,
        "observed": observed, "qk_indices": index_sets, "qk_widths": sorted(widths),
        "active_replacements": active, "intervention_calls": intervention["calls"],
        "mean_n": mean_n, "identity": identity, "fit_diagnostics": diagnostics,
        "run_bridge": {"L2CF": C.SEL.get("L2CF"), "runtime_s": run.get("runtime_s")},
    }
    torch.save(payload, CONDITION_FILES[condition])
    print(f"RUNG453 child {condition} saved {CONDITION_FILES[condition].name}", flush=True)


def parent() -> None:
    started = time.time()
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    existing = [path.exists() for path in CONDITION_FILES.values()]
    if OUT.exists() or BUNDLE.exists() or (any(existing) and not all(existing)):
        raise RuntimeError("rung453 output namespace is occupied or partial")
    bank = {row["candidate_id"]: row for row in json.loads(BANK.read_text())["rows"]}
    if any(bank[name]["price_scalars"] != PRICES[name] or bank[name]["family_role"] != "teaching"
           for name in IDS):
        raise RuntimeError("MLP-PCA bank prices/roles changed")
    if all(existing):
        print("RUNG453 scorer resume: reusing three completed condition bundles", flush=True)
    else:
        source = ROOT / "ops/simplicity_mlp_pca_independent_reliability.py"
        for condition in CONDITIONS:
            environment = dict(os.environ); environment.pop("BQLIB_DRYRUN", None)
            environment["RUNG453_CHILD"] = condition
            subprocess.run([sys.executable, str(source)], env=environment, check=True)

    data = {name: torch.load(path, map_location="cpu", weights_only=True)
            for name, path in CONDITION_FILES.items()}
    native_aux = data["unablated"]["native_aux"]
    native = native_aux["native_ce"].float(); native_ko = native_aux["native_ko_ce"].float()
    partner = native_aux["partner_ce"].float()
    replay_max = float((native_aux["native_replay_ce"].float() - native).abs().max())
    all_index = torch.arange(native.numel())
    waves = [torch.arange(0, 96 * TOKENS), torch.arange(96 * TOKENS, 192 * TOKENS)]
    arms: dict[str, Any] = {}; bundle_arms: dict[str, Any] = {}
    for name in IDS:
        p = data["unablated"]["candidate_ce"][name].float()
        pko = data["knockout"]["candidate_ce"][name].float()
        pq = data["partner"]["candidate_ce"][name].float()
        arms[name] = {"price_scalars": PRICES[name],
                      "full": metrics(native, native_ko, p, pko, partner, pq, all_index),
                      "waves": [metrics(native, native_ko, p, pko, partner, pq, index)
                                for index in waves]}
        bundle_arms[name] = {"candidate_ce": p.half(), "candidate_ko_ce": pko.half(),
                             "candidate_partner_ce": pq.half()}

    spec = json.loads(SPEC.read_text()); old = json.loads(OLD_RESULT.read_text())
    old_values = {metric: [old["arms"][name]["full"][f"{metric}_normalized_error"] for name in IDS]
                  for metric in ("removal", "composition")}
    new_values = {metric: [arms[name]["full"][f"{metric}_normalized_error"] for name in IDS]
                  for metric in ("removal", "composition")}
    magnitude = {metric: {"pearson": pearson(old_values[metric], new_values[metric]),
                          "mean_absolute_shift": sum(abs(a - b) for a, b in
                                                     zip(old_values[metric], new_values[metric])) / len(IDS)}
                 for metric in ("removal", "composition")}
    full_maps = {metric: {name: arms[name]["full"][f"{metric}_normalized_error"] for name in IDS}
                 for metric in ("removal", "composition")}
    pooled_pairs = {metric: pair_accuracy(full_maps[metric], spec["separated_pairs"][metric])
                    for metric in ("removal", "composition")}
    wave_pairs = []
    for wave in range(2):
        maps = {metric: {name: arms[name]["waves"][wave][f"{metric}_normalized_error"] for name in IDS}
                for metric in ("removal", "composition")}
        wave_pairs.append({metric: pair_accuracy(maps[metric], spec["separated_pairs"][metric])
                           for metric in ("removal", "composition")})
    ladder = ("mlp_pca_p8_17_r256", "mlp_pca_p8_17_r384", "mlp_pca_p8_17_r512")
    ladder_values = {
        "full": {metric: [full_maps[metric][name] for name in ladder]
                 for metric in ("removal", "composition")},
        "wave0": {metric: [arms[name]["waves"][0][f"{metric}_normalized_error"] for name in ladder]
                  for metric in ("removal", "composition")},
        "wave1": {metric: [arms[name]["waves"][1][f"{metric}_normalized_error"] for name in ladder]
                  for metric in ("removal", "composition")},
    }
    ladder_monotone = {
        scope: {metric: all(values[index] > values[index + 1] for index in range(2))
                for metric, values in values_by_metric.items()}
        for scope, values_by_metric in ladder_values.items()
    }
    identity = all(item["identity"] for item in data.values()) and all(
        data[name]["observed"] == data["unablated"]["observed"]
        and data[name]["active_replacements"] == data["unablated"]["active_replacements"]
        for name in CONDITIONS)
    live = bool(data["knockout"]["intervention_calls"] == len(IDS) * 48
                and data["partner"]["intervention_calls"] == len(IDS) * 48
                and data["unablated"]["native_counts"] == {"knockout": 48, "partner": 48}
                and all(data[name]["mean_n"] == 128 * TOKENS for name in CONDITIONS))
    pred_a = bool(identity and live and replay_max == 0 and not spec["outcome_access"]["sealed_opened"])
    pred_b = bool(all(magnitude[metric]["pearson"] >= .85
                      and magnitude[metric]["mean_absolute_shift"] <= .04
                      for metric in ("removal", "composition")))
    pred_c = bool(pooled_pairs["removal"]["accuracy"] >= .85
                  and pooled_pairs["composition"]["accuracy"] >= .85
                  and all(ladder_monotone["full"].values()))
    pred_d = bool(all(wave_pairs[wave][metric]["accuracy"] >= .75
                      for wave in range(2) for metric in ("removal", "composition"))
                  and all(value for scope in ("wave0", "wave1")
                          for value in ladder_monotone[scope].values()))
    candidate_live = all(abs(arms[name]["full"]["candidate_ce_damage"]) >= 1e-4 for name in IDS)
    strong_null = bool(not pred_a or (native_ko - native).norm() <= 1e-6
                       or abs(float((partner - native).mean())) < 1e-4 or not candidate_live
                       or any(magnitude[metric]["pearson"] < 0 for metric in ("removal", "composition"))
                       or any(pooled_pairs[metric]["accuracy"] < .60 for metric in ("removal", "composition")))
    bundle = {"schema": "simplicity_mlp_pca_independent_reliability_bundle_v1",
              "native_ce": native.half(), "native_ko_ce": native_ko.half(),
              "partner_ce": partner.half(), "arms": bundle_arms}
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 453,
        "claim_level": "independent_complete_candidate_consequence_reliability",
        "sealed_opened": False, "condition_files": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in CONDITION_FILES.items()},
        "complete_identity_all_conditions": identity, "native_replay_max_abs": replay_max,
        "intervention_counts": {name: data[name]["intervention_calls"] for name in CONDITIONS},
        "native_counts": data["unablated"]["native_counts"], "arms": arms,
        "old_to_independent_magnitude": magnitude, "pooled_pair_reproduction": pooled_pairs,
        "wave_pair_reproduction": wave_pairs, "rank_ladder": list(ladder),
        "rank_ladder_monotone": ladder_monotone,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        'pred_a_complete_instrument': pred_a,
        'pred_b_continuous_magnitudes_reproduce': pred_b,
        'pred_c_uncertainty_separated_pairs_reproduce': pred_c,
        'pred_d_independent_wave_reliability': pred_d,
        "strong_null_complete_family_unusable": strong_null,
        "next_step": ("count_mlp_pca_family_and_build_vocabulary_teaching_family"
                      if pred_a and pred_b and pred_c and pred_d and not strong_null
                      else "do_not_count_mlp_pca_family_build_vocabulary_teaching_family"),
        "new_deployed_values": 0, "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() and sha256(path) == digest for path, digest in HASHES.items())
        spec = json.loads(SPEC.read_text())
        assert len(spec["separated_pairs"]["removal"]) == 13
        assert len(spec["separated_pairs"]["composition"]) == 16
        assert tuple(spec["candidate_ids"]) == IDS and set(IDS) == set(PRICES)
        print("RUNG453 INDEPENDENT MLP-PCA RELIABILITY | dry run: hashes, rows, pairs, bars valid")
        return
    mode = os.environ.get("RUNG453_CHILD")
    if mode:
        child(mode)
    else:
        parent()


if __name__ == "__main__":
    main()
