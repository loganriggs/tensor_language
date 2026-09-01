"""RUNG448 -- PROSPECTIVE MLP0 REMOVAL/COMPOSITION TEACHING LABELS.

pred_a: frozen hashes, exact ranks, native replay, and dispatch liveness hold.
pred_b: rank orders both removal and composition errors at Spearman >= .50.
pred_c: both normalized-error spans are >= .015.
pred_d: two-wave ordering Spearman is >= .70 for both labels.
Strong null: dead intervention/arms, both spans < .003, negative wave stability, or invalid instrument.
Label generation only; no sealed-family, predictor, compression, or adoption claim.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
OUT = ROOT / "simplicity_mlp0_teaching_consequences_results.json"
BUNDLE = ROOT / "simplicity_mlp0_teaching_consequences_bundle.pt"
BANK = POLY / "prospective_consequence_candidate_bank_v1.json"
ROWS_RECEIPT = ROOT / "simplicity_consequence_v1_rows_receipt.json"
ROWS = ROOT / ".rowcache_simplicity_consequence_v1/teaching_96.pt"
FIT = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FACTOR = ROOT / "mlp16_rank2_quadratic_factored.pt"
RANKS = (256, 384, 448, 512, 640)
HASHES = {
    BANK: "e35d5c0aa1dae34173b93ae4d81cafa8317539adfaf7c74bfe7decb068ac47be",
    ROWS_RECEIPT: "1611c5bd60491a6b600950874ae55cd5925afad12096a48de3426e88e9cfc5d8",
    ROWS: "b94fb82be422e17411ed8ebf1b3e94956848e074687dbdf593ae9285da837014",
    FIT: "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
    FACTOR: "b9870f738b528e988ff9a1e04cdc6e1096de8ab0dc5fa86bb76229812d9ffb6e",
}
D = 1152
H = 4608


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.double()[None], right.double()[None]))


def rankdata(value: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(value, stable=True)
    ranks = torch.empty(len(value), dtype=torch.float64)
    ranks[order] = torch.arange(len(value), dtype=torch.float64)
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    a = rankdata(torch.tensor(left, dtype=torch.float64)); b = rankdata(torch.tensor(right, dtype=torch.float64))
    a -= a.mean(); b -= b.mean()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def metric_slice(native: torch.Tensor, ko: torch.Tensor, p: torch.Tensor, pko: torch.Tensor,
                 q: torch.Tensor, pq: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    rn = (ko - native)[indices].double(); rp = (pko - p)[indices].double()
    physical = (pq - native)[indices].double()
    additive = ((p - native) + (q - native))[indices].double()
    return {
        "removal_cosine": cosine(rp, rn),
        "removal_normalized_error": float((rp - rn).norm() / rn.norm().clamp_min(1e-12)),
        "removal_norm_ratio": float(rp.norm() / rn.norm().clamp_min(1e-12)),
        "composition_cosine": cosine(physical, additive),
        "composition_normalized_error": float((physical - additive).norm() / additive.norm().clamp_min(1e-12)),
        "composition_interaction_mean": float((physical - additive).mean()),
        "candidate_ce_damage": float((p - native)[indices].mean()),
        "partner_ce_damage": float((q - native)[indices].mean()),
        "physical_composition_ce_damage": float((pq - native)[indices].mean()),
    }


@torch.no_grad()
def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() and sha256(path) == digest for path, digest in HASHES.items())
        bank = json.loads(BANK.read_text())
        ids = {row["candidate_id"] for row in bank["rows"] if row["family_role"] == "teaching"}
        assert {f"mlp0_context_input_r{rank}" for rank in RANKS} <= ids
        assert len(RANKS) == 5 and 14_984 == 4 * D + 4 * 2 * D + 4 * 2 + D
        print("RUNG448 MLP0 TEACHING CONSEQUENCES | dry run: hashes, arms, bars valid")
        return

    started = time.time()
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    if receipt["entries"]["TEACHING"]["tensor_sha256"] != "660346718863e1abda13fd24c5e0f96bab5c536bd3090eb6a3ae64f7b53d2d1c":
        raise RuntimeError("TEACHING semantic tensor hash changed")

    sys.path.insert(0, str(ROOT.parents[1]))
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(ROOT / "ops"))
    from bilin18_observed_model_facade import load_bilin18, forward_with_dispatch
    from mlp_late_context_metric_shared_input_screen import _rrr_program

    model, checkpoint = load_bilin18(device="cuda", dtype=torch.float32)
    rows = torch.load(ROWS, map_location="cpu", weights_only=True).long()
    fit_value = torch.load(FIT, map_location="cpu", weights_only=True)
    fit_value = fit_value["rows"] if isinstance(fit_value, dict) else fit_value
    fit_rows = fit_value[:24, :257].long()
    calibration_rows = fit_value[:128, :257].long()
    factor_cpu = torch.load(FACTOR, map_location="cpu", weights_only=True)
    factor = {key: value.cuda().float() for key, value in factor_cpu.items()}
    expected_factor = {"output_directions": [4, D], "form_vectors": [4, 2, D],
                       "form_values": [4, 2], "constant": [D]}
    if {key: list(value.shape) for key, value in factor_cpu.items()} != expected_factor or sum(
        value.numel() for value in factor_cpu.values()) != 14_984:
        raise RuntimeError("physical MLP16 factor identity changed")

    def native_attention(event):
        return event.block.attn(event.state, event.first_value)

    covariance = torch.zeros(D, D, device="cuda", dtype=torch.float64)
    covariance_n = 0

    def covariance_mlp(event):
        nonlocal covariance_n
        if event.site == 0:
            value = event.state.detach().double().reshape(-1, D)
            covariance.addmm_(value.T, value); covariance_n += value.shape[0]
        return event.block.mlp(event.state)

    for start in range(0, len(fit_rows), 4):
        forward_with_dispatch(model, fit_rows[start:start + 4, :-1].cuda(), native_attention, covariance_mlp)
    covariance = (covariance / covariance_n).float()
    covariance = 0.5 * (covariance + covariance.T)
    programs: dict[int, dict[str, torch.Tensor]] = {}
    diagnostics: dict[str, Any] = {}
    for rank in RANKS:
        program, _basis, diagnostic = _rrr_program(model.transformer.h[0].mlp, covariance, rank=rank)
        programs[rank] = {key: value.detach().cuda().float() for key, value in program.items()}
        diagnostics[str(rank)] = diagnostic
    del covariance

    mean_sum = torch.zeros(D, device="cuda", dtype=torch.float64); mean_n = 0

    def mean_attention(event):
        nonlocal mean_n
        output, next_value = event.block.attn(event.state, event.first_value)
        if event.site == 16:
            flat = output.detach().double().reshape(-1, D)
            mean_sum.add_(flat.sum(0)); mean_n += flat.shape[0]
        return output, next_value

    def native_mlp(event):
        return event.block.mlp(event.state)

    for start in range(0, len(calibration_rows), 4):
        forward_with_dispatch(model, calibration_rows[start:start + 4, :-1].cuda(), mean_attention, native_mlp)
    mean_value = (mean_sum / mean_n).float()

    counts = {"candidate": 0, "partner": 0, "knockout": 0}

    def factored(x: torch.Tensor) -> torch.Tensor:
        projections = torch.einsum("...d,rkd->...rk", x.float(), factor["form_vectors"])
        coefficients = (projections.square() * factor["form_values"]).sum(-1)
        return coefficients @ factor["output_directions"] + factor["constant"]

    def score(*, program=None, partner=False, knockout=False) -> torch.Tensor:
        values = []
        def attention(event):
            output, next_value = event.block.attn(event.state, event.first_value)
            if knockout and event.site == 16:
                counts["knockout"] += 1
                output = mean_value.expand_as(output).to(output.dtype)
            return output, next_value
        def mlp(event):
            if program is not None and event.site == 0:
                counts["candidate"] += 1
                encoder, left, right = program["encoder"], program["left"], program["right"]
                z = event.state.float() @ encoder.T
                hidden = (z @ left.T) * (z @ right.T)
                return (hidden @ program["down"].T + program["bias"]).to(event.state.dtype)
            if partner and event.site == 16:
                counts["partner"] += 1
                return factored(event.state).to(event.state.dtype)
            return event.block.mlp(event.state)
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4].cuda()
            logits = forward_with_dispatch(model, batch[:, :-1], attention, mlp)
            values.append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                          batch[:, 1:].reshape(-1), reduction="none").cpu())
        return torch.cat(values).float()

    native = score(); native_replay = score()
    native_replay_max = float((native_replay - native).abs().max())
    native_ko = score(knockout=True)
    partner = score(partner=True)
    native_effect = native_ko - native
    magnitude = native_effect.abs()
    target_mask = magnitude >= torch.quantile(magnitude, .75)
    collateral_mask = magnitude <= torch.quantile(magnitude, .50)
    all_index = torch.arange(native.numel())
    wave_indices = [torch.arange(0, 48 * 256), torch.arange(48 * 256, 96 * 256)]

    arms: dict[str, Any] = {}
    bundle_arms: dict[str, Any] = {}
    for rank in RANKS:
        p = score(program=programs[rank])
        pko = score(program=programs[rank], knockout=True)
        pq = score(program=programs[rank], partner=True)
        full = metric_slice(native, native_ko, p, pko, partner, pq, all_index)
        full["target_removal_normalized_error"] = float(
            ((pko - p - native_effect)[target_mask]).norm()
            / native_effect[target_mask].norm().clamp_min(1e-12))
        full["collateral_removal_rms_error"] = float(
            (pko - p - native_effect)[collateral_mask].double().square().mean().sqrt())
        waves = [metric_slice(native, native_ko, p, pko, partner, pq, index) for index in wave_indices]
        arms[str(rank)] = {"rank": rank, "full": full, "waves": waves,
                           "price_scalars": next(row["price_scalars"] for row in json.loads(BANK.read_text())["rows"]
                                                  if row["candidate_id"] == f"mlp0_context_input_r{rank}")}
        bundle_arms[str(rank)] = {"candidate_ce": p.half(), "candidate_ko_ce": pko.half(),
                                  "candidate_partner_ce": pq.half()}

    removal = [arms[str(rank)]["full"]["removal_normalized_error"] for rank in RANKS]
    composition = [arms[str(rank)]["full"]["composition_normalized_error"] for rank in RANKS]
    removal_span = max(removal) - min(removal); composition_span = max(composition) - min(composition)
    rank_removal_rho = spearman(list(RANKS), [-value for value in removal])
    rank_composition_rho = spearman(list(RANKS), [-value for value in composition])
    removal_wave_rho = spearman(
        [arms[str(rank)]["waves"][0]["removal_normalized_error"] for rank in RANKS],
        [arms[str(rank)]["waves"][1]["removal_normalized_error"] for rank in RANKS])
    composition_wave_rho = spearman(
        [arms[str(rank)]["waves"][0]["composition_normalized_error"] for rank in RANKS],
        [arms[str(rank)]["waves"][1]["composition_normalized_error"] for rank in RANKS])
    shapes_valid = all(programs[rank]["encoder"].shape == (rank, D) for rank in RANKS)
    live = counts["candidate"] == len(RANKS) * 3 * 24 and counts["partner"] == 6 * 24 \
        and counts["knockout"] == 6 * 24 and mean_n == 128 * 256 and covariance_n == 24 * 256
    arms_live = all(abs(arms[str(rank)]["full"]["candidate_ce_damage"]) >= 1e-4 for rank in RANKS)
    pred_a = bool(native_replay_max == 0 and shapes_valid and live)
    pred_b = bool(rank_removal_rho >= .50 and rank_composition_rho >= .50)
    pred_c = bool(removal_span >= .015 and composition_span >= .015)
    pred_d = bool(removal_wave_rho >= .70 and composition_wave_rho >= .70)
    strong_null = bool(not pred_a or native_effect.norm() <= 1e-6 or abs(float((partner-native).mean())) < 1e-4
                       or not arms_live or (removal_span < .003 and composition_span < .003)
                       or removal_wave_rho < 0 or composition_wave_rho < 0)

    bundle = {"schema": "simplicity_mlp0_teaching_consequences_bundle_v1",
              "native_ce": native.half(), "native_ko_ce": native_ko.half(), "partner_ce": partner.half(),
              "target_mask": target_mask, "collateral_mask": collateral_mask, "arms": bundle_arms}
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 448, "claim_level": "prospective_teaching_label_generation",
        "checkpoint": checkpoint.__dict__, "rows_receipt_sha256": HASHES[ROWS_RECEIPT],
        "teaching_tensor_sha256": receipt["entries"]["TEACHING"]["tensor_sha256"],
        "sealed_opened": False, "fit_cache_sha256": HASHES[FIT], "factor_sha256": HASHES[FACTOR],
        "program_diagnostics": diagnostics, "dispatch_counts": counts,
        "native_replay_max_abs": native_replay_max, "native_effect_norm": float(native_effect.norm()),
        "target_positions": int(target_mask.sum()), "collateral_positions": int(collateral_mask.sum()),
        "arms": arms, "removal_error_span": removal_span, "composition_error_span": composition_span,
        "rank_removal_spearman": rank_removal_rho, "rank_composition_spearman": rank_composition_rho,
        "removal_wave_order_spearman": removal_wave_rho,
        "composition_wave_order_spearman": composition_wave_rho,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        'pred_a_instrument': pred_a, 'pred_b_rank_orders_both': pred_b,
        'pred_c_labels_vary': pred_c, 'pred_d_wave_stability': pred_d,
        "strong_null_family_unusable": strong_null,
        "next_step": "count_mlp0_family_and_build_next_teaching_family" if pred_a and not strong_null
                     else "do_not_count_mlp0_family_repair_instrument_or_change_family",
        "new_deployed_values": 0, "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
