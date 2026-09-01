"""RUNG449 -- COMPLETE MIXED104+MLP0 PROSPECTIVE TEACHING CONSEQUENCES.

Corrects rung448's local-object/full-price mismatch by installing the whole
historical compiled candidate for every consequence.  Rung448 remains a valid
local diagnostic and is never relabeled.
"""

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


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
OUT = ROOT / "simplicity_mlp0_complete_candidate_consequences_results.json"
BUNDLE = ROOT / "simplicity_mlp0_complete_candidate_consequences_bundle.pt"
ROWS = ROOT / ".rowcache_simplicity_consequence_v1/teaching_96.pt"
ROWS_RECEIPT = ROOT / "simplicity_consequence_v1_rows_receipt.json"
BANK = POLY / "prospective_consequence_candidate_bank_v1.json"
FIT = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FACTOR = ROOT / "mlp16_rank2_quadratic_factored.pt"
LOCAL = ROOT / "simplicity_mlp0_teaching_consequences_bundle.pt"
CONDITIONS = ("unablated", "knockout", "partner")
CONDITION_FILES = {name: ROOT / f"simplicity_mlp0_complete_{name}.pt" for name in CONDITIONS}
RANKS = (256, 384, 448, 512, 640)
PRICES = {256: 531_632_438, 384: 532_959_542, 448: 533_623_094,
          512: 534_286_646, 640: 535_613_750}
D = 1152
HASHES = {
    BANK: "e35d5c0aa1dae34173b93ae4d81cafa8317539adfaf7c74bfe7decb068ac47be",
    ROWS_RECEIPT: "1611c5bd60491a6b600950874ae55cd5925afad12096a48de3426e88e9cfc5d8",
    ROWS: "b94fb82be422e17411ed8ebf1b3e94956848e074687dbdf593ae9285da837014",
    FIT: "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
    FACTOR: "b9870f738b528e988ff9a1e04cdc6e1096de8ab0dc5fa86bb76229812d9ffb6e",
    LOCAL: "0c95e125a712c5c2f4aae5186e0acc2211f6dba586146b13765c01d3e9a8360f",
    ROOT / "ops/cevdump_ct96.py": "1fc1d2a405b94228885921b6085294a7ada609badc6c4c834c92f447d483c932",
    ROOT / "ops/mixed104_mlp0_context_metric_lower_rank_frontier.py":
        "3c89d48ca99b8e18f5bb99b78cdb07ed71e85aded057279374ba7a193991a6d9",
    ROOT / "ops/mixed104_mlp0_context_metric_input_frontier.py":
        "25eb8c265dcbb43d50e9648768ae4bc862f0575793a08ae422a5693d007e6ebc",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def rankdata(value: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(value, stable=True)
    ranks = torch.empty(len(value), dtype=torch.float64)
    ranks[order] = torch.arange(len(value), dtype=torch.float64)
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    a = rankdata(torch.tensor(left, dtype=torch.float64))
    b = rankdata(torch.tensor(right, dtype=torch.float64))
    a -= a.mean(); b -= b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.double()[None], right.double()[None]))


def metrics(native: torch.Tensor, ko: torch.Tensor, p: torch.Tensor, pko: torch.Tensor,
            q: torch.Tensor, pq: torch.Tensor, index: torch.Tensor) -> dict[str, float]:
    rn = (ko - native)[index].double(); rp = (pko - p)[index].double()
    physical = (pq - native)[index].double()
    additive = ((p - native) + (q - native))[index].double()
    return {
        "removal_cosine": cosine(rp, rn),
        "removal_normalized_error": float((rp - rn).norm() / rn.norm().clamp_min(1e-12)),
        "removal_norm_ratio": float(rp.norm() / rn.norm().clamp_min(1e-12)),
        "composition_cosine": cosine(physical, additive),
        "composition_normalized_error": float(
            (physical - additive).norm() / additive.norm().clamp_min(1e-12)),
        "composition_interaction_mean": float((physical - additive).mean()),
        "candidate_ce_damage": float((p - native)[index].mean()),
        "partner_ce_damage": float((q - native)[index].mean()),
        "physical_composition_ce_damage": float((pq - native)[index].mean()),
    }


def factored(value: torch.Tensor, program: dict[str, torch.Tensor]) -> torch.Tensor:
    projection = torch.einsum("...d,rkd->...rk", value.float(), program["form_vectors"])
    coefficient = (projection.square() * program["form_values"]).sum(-1)
    return coefficient @ program["output_directions"] + program["constant"]


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
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    C.m.eval()
    for parameter in C.m.parameters():
        parameter.requires_grad_(False)
    validate_production_model(C.m)
    rows = torch.load(ROWS, map_location="cpu", weights_only=True).long()
    cached = torch.load(FIT, map_location="cpu", weights_only=True)
    cached = cached["rows"] if isinstance(cached, dict) else cached
    fit_rows = cached[:24, :257].long().contiguous()
    calibration_rows = cached[:128, :257].long().contiguous()
    covariance = _covariance(C.m, fit_rows, _manual_logits)
    variants: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    diagnostics: dict[str, Any] = {}
    for rank in RANKS:
        program, _basis, diagnostic = _rrr_program(C.m.transformer.h[0].mlp,
                                                   covariance, rank=rank)
        variants[f"r{rank}"] = {0: {key: value.cpu() for key, value in program.items()}}
        diagnostics[str(rank)] = diagnostic
    del covariance
    torch.cuda.empty_cache()

    mean_sum = torch.zeros(D, device="cuda", dtype=torch.float64); mean_n = 0
    def capture_mean(_module, _args, output):
        nonlocal mean_n
        value = output[0].detach().double().reshape(-1, D)
        mean_sum.add_(value.sum(0)); mean_n += value.shape[0]
    handle = C.m.transformer.h[16].attn.register_forward_hook(capture_mean)
    try:
        for start in range(0, len(calibration_rows), 4):
            _manual_logits(C.m, calibration_rows[start:start + 4, :-1].cuda())
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

    C.CROWS, C.CBASE, C.NFLAT = rows, torch.zeros(rows.numel() - len(rows)), 96 * 256
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_input_programs": variants["r256"],
        "final_mlp_input_program_variants": variants,
        "final_mlp_input_primary_variant": "r256",
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

    cevs = C.SEL.get("_final_mlp_input_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_input_variant_observed", {})
    wanted = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(item[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for item in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    identity = bool(
        set(cevs) == {f"r{rank}" for rank in RANKS}
        and all({int(key): int(value) for key, value in observed[f"r{rank}"].items()}
                == {0: rank} for rank in RANKS)
        and set(index_sets) == set(range(2, 18))
        and all(value == wanted for value in index_sets.values()) and widths == {104}
        and not any(name in active for name in ("a0", "a1v", "tailE")))
    if not identity:
        raise RuntimeError("complete mixed104 candidate identity changed")
    if condition in ("knockout", "partner") and intervention["calls"] != len(RANKS) * 24:
        raise RuntimeError(f"{condition} final-call count changed: {intervention['calls']}")

    payload = {
        "schema": "simplicity_mlp0_complete_condition_v1", "condition": condition,
        "candidate_ce": {str(rank): cevs[f"r{rank}"].float().cpu() for rank in RANKS},
        "native_aux": native_aux, "native_counts": native_counts, "mean_n": mean_n,
        "fit_diagnostics": diagnostics, "observed": observed, "qk_indices": index_sets,
        "qk_widths": sorted(widths), "active_replacements": active,
        "intervention_calls": intervention["calls"], "identity": identity,
        "run_bridge": {"L2CF": C.SEL.get("L2CF"), "runtime_s": run.get("runtime_s")},
    }
    torch.save(payload, CONDITION_FILES[condition])
    print(f"RUNG449 child {condition} saved {CONDITION_FILES[condition].name}", flush=True)


def parent() -> None:
    started = time.time()
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    if any(path.exists() for path in CONDITION_FILES.values()) or OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung449 output namespace already exists; preserve and reregister before rerun")
    bank = json.loads(BANK.read_text())
    bank_rows = {row["candidate_id"]: row for row in bank["rows"]}
    for rank in RANKS:
        row = bank_rows[f"mlp0_context_input_r{rank}"]
        if row["price_scalars"] != PRICES[rank] or row["family_role"] != "teaching":
            raise RuntimeError("frozen bank price/role changed")

    for condition in CONDITIONS:
        environment = dict(os.environ)
        environment.pop("BQLIB_DRYRUN", None)
        environment["RUNG449_CHILD"] = condition
        source = ROOT / "ops/simplicity_mlp0_complete_candidate_consequences.py"
        subprocess.run([sys.executable, str(source)], env=environment, check=True)

    data = {name: torch.load(path, map_location="cpu", weights_only=True)
            for name, path in CONDITION_FILES.items()}
    base = data["unablated"]["native_aux"]
    native = base["native_ce"].float(); native_ko = base["native_ko_ce"].float()
    partner = base["partner_ce"].float()
    replay_max = float((base["native_replay_ce"].float() - native).abs().max())
    local = torch.load(LOCAL, map_location="cpu", weights_only=True)["arms"]
    all_index = torch.arange(native.numel())
    waves = [torch.arange(0, 48 * 256), torch.arange(48 * 256, 96 * 256)]
    native_effect = native_ko - native
    magnitude = native_effect.abs()
    target = magnitude >= torch.quantile(magnitude, .75)
    collateral = magnitude <= torch.quantile(magnitude, .50)

    arms: dict[str, Any] = {}; full_bundle: dict[str, Any] = {}
    for rank in RANKS:
        key = str(rank)
        p = data["unablated"]["candidate_ce"][key].float()
        pko = data["knockout"]["candidate_ce"][key].float()
        pq = data["partner"]["candidate_ce"][key].float()
        full = metrics(native, native_ko, p, pko, partner, pq, all_index)
        full["target_removal_normalized_error"] = float(
            ((pko - p - native_effect)[target]).norm()
            / native_effect[target].norm().clamp_min(1e-12))
        full["collateral_removal_rms_error"] = float(
            (pko - p - native_effect)[collateral].double().square().mean().sqrt())
        full["complete_vs_local_candidate_ce_mean_abs"] = float(
            (p - local[key]["candidate_ce"].float()).abs().mean())
        arms[key] = {"rank": rank, "price_scalars": PRICES[rank], "full": full,
                     "waves": [metrics(native, native_ko, p, pko, partner, pq, index)
                               for index in waves]}
        full_bundle[key] = {"candidate_ce": p.half(), "candidate_ko_ce": pko.half(),
                            "candidate_partner_ce": pq.half()}

    removal = [arms[str(rank)]["full"]["removal_normalized_error"] for rank in RANKS]
    composition = [arms[str(rank)]["full"]["composition_normalized_error"] for rank in RANKS]
    removal_span = max(removal) - min(removal); composition_span = max(composition) - min(composition)
    rank_removal = spearman(list(RANKS), [-value for value in removal])
    rank_composition = spearman(list(RANKS), [-value for value in composition])
    removal_wave = spearman(
        [arms[str(rank)]["waves"][0]["removal_normalized_error"] for rank in RANKS],
        [arms[str(rank)]["waves"][1]["removal_normalized_error"] for rank in RANKS])
    composition_wave = spearman(
        [arms[str(rank)]["waves"][0]["composition_normalized_error"] for rank in RANKS],
        [arms[str(rank)]["waves"][1]["composition_normalized_error"] for rank in RANKS])
    same_identity = all(item["identity"] for item in data.values()) and all(
        data[name]["observed"] == data["unablated"]["observed"]
        and data[name]["active_replacements"] == data["unablated"]["active_replacements"]
        for name in CONDITIONS)
    complete_differs = all(
        arms[str(rank)]["full"]["complete_vs_local_candidate_ce_mean_abs"] >= 1e-4
        for rank in RANKS)
    live = bool(data["knockout"]["intervention_calls"] == 120
                and data["partner"]["intervention_calls"] == 120
                and base["native_counts"] == {"knockout": 24, "partner": 24}
                and data["unablated"]["mean_n"] == 128 * 256)
    pred_a = bool(replay_max == 0 and same_identity and complete_differs and live)
    pred_b = bool(rank_removal >= .50 and rank_composition >= .50)
    pred_c = bool(removal_span >= .015 and composition_span >= .015)
    pred_d = bool(removal_wave >= .70 and composition_wave >= .70)
    candidate_live = all(abs(arms[str(rank)]["full"]["candidate_ce_damage"]) >= 1e-4
                         for rank in RANKS)
    strong_null = bool(not pred_a or native_effect.norm() <= 1e-6
                       or abs(float((partner - native).mean())) < 1e-4 or not candidate_live
                       or (removal_span < .003 and composition_span < .003)
                       or removal_wave < 0 or composition_wave < 0)

    bundle = {"schema": "simplicity_mlp0_complete_candidate_bundle_v1",
              "native_ce": native.half(), "native_ko_ce": native_ko.half(),
              "partner_ce": partner.half(), "target_mask": target,
              "collateral_mask": collateral, "arms": full_bundle}
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 449,
        "claim_level": "prospective_complete_candidate_teaching_label_generation",
        "checkpoint": {"revision": "ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240",
                       "weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"},
        "sealed_opened": False, "rung448_local_receipt_preserved": True,
        "condition_files": {name: {"path": str(path), "sha256": sha256(path)}
                            for name, path in CONDITION_FILES.items()},
        "native_replay_max_abs": replay_max, "native_effect_norm": float(native_effect.norm()),
        "complete_identity_all_conditions": same_identity, "complete_differs_from_local": complete_differs,
        "intervention_counts": {name: data[name]["intervention_calls"] for name in CONDITIONS},
        "native_aux_counts": base["native_counts"], "arms": arms,
        "removal_error_span": removal_span, "composition_error_span": composition_span,
        "rank_removal_spearman": rank_removal, "rank_composition_spearman": rank_composition,
        "removal_wave_order_spearman": removal_wave,
        "composition_wave_order_spearman": composition_wave,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        'pred_a_complete_instrument': pred_a,
        'pred_b_rank_orders_both': pred_b,
        'pred_c_labels_vary': pred_c,
        'pred_d_wave_stability': pred_d,
        "strong_null_complete_family_unusable": strong_null,
        "next_step": ("count_complete_mlp0_family_and_build_mlp_pca_teaching_family"
                      if pred_a and pred_b and pred_c and pred_d and not strong_null
                      else "do_not_count_complete_mlp0_family"),
        "new_deployed_values": 0, "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() and sha256(path) == digest for path, digest in HASHES.items())
        assert len(RANKS) == 5 and set(PRICES) == set(RANKS)
        assert not (ROOT / ".rowcache_simplicity_consequence_v1/sealed_confirmation_96.pt") in HASHES
        print("RUNG449 COMPLETE MLP0 CONSEQUENCES | dry run: hashes, object, arms, bars valid")
        return
    mode = os.environ.get("RUNG449_CHILD")
    if mode:
        child(mode)
    else:
        parent()


if __name__ == "__main__":
    main()
