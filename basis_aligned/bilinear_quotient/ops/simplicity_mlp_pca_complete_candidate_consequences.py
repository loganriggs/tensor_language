"""RUNG450 -- COMPLETE MIXED104 MLP-PCA TEACHING CONSEQUENCES."""

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

from receipt import dump
from simplicity_mlp0_complete_candidate_consequences import factored, metrics, spearman


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
OUT = ROOT / "simplicity_mlp_pca_complete_candidate_consequences_results.json"
BUNDLE = ROOT / "simplicity_mlp_pca_complete_candidate_consequences_bundle.pt"
ROWS = ROOT / ".rowcache_simplicity_consequence_v1/teaching_96.pt"
ROWS_RECEIPT = ROOT / "simplicity_consequence_v1_rows_receipt.json"
BANK = POLY / "prospective_consequence_candidate_bank_v1.json"
GENERIC_FIT = ROOT / ".rowcache/fineweb_n480_skip80.pt"
MEAN_FIT = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
CENSUS = ROOT / "census_state_diverse.pt"
FACTOR = ROOT / "mlp16_rank2_quadratic_factored.pt"
NATIVE = ROOT / "simplicity_mlp0_complete_unablated.pt"
CONDITIONS = ("unablated", "knockout", "partner")
CONDITION_FILES = {name: ROOT / f"simplicity_mlp_pca_complete_{name}.pt" for name in CONDITIONS}
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
HASHES = {
    BANK: "e35d5c0aa1dae34173b93ae4d81cafa8317539adfaf7c74bfe7decb068ac47be",
    ROWS_RECEIPT: "1611c5bd60491a6b600950874ae55cd5925afad12096a48de3426e88e9cfc5d8",
    ROWS: "b94fb82be422e17411ed8ebf1b3e94956848e074687dbdf593ae9285da837014",
    GENERIC_FIT: "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496",
    MEAN_FIT: "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
    CENSUS: "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
    FACTOR: "b9870f738b528e988ff9a1e04cdc6e1096de8ab0dc5fa86bb76229812d9ffb6e",
    NATIVE: "e3fa3c373b11ea455bf843cb555e6beb4cd68451bdda97f82a8393692096dd59",
    ROOT / "ops/cevdump_ct96.py": "1fc1d2a405b94228885921b6085294a7ada609badc6c4c834c92f447d483c932",
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


@torch.no_grad()
def build_variants(C, manual_logits):
    import census_lib as CN
    import mlp_activation_pca_four_layer_composition as PCA
    import mixed104_pca_fixed_pair_rank_frontier as RANK
    import mixed104_pca_certificate_gradient_hybrid as GRAD

    generic_rows = PCA._load_rows(GENERIC_FIT, 16)
    pca256 = PCA._fit_pca(PCA._capture_outputs(C.m, generic_rows, manual_logits))
    rankfit = RANK._fit_selected_pca(C.m, generic_rows, manual_logits)

    CN.use_state(str(CENSUS))
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    tags = sorted(tag for tag in battery if tag in set(CN.all_tags()))
    fit_tags = tags[::2]
    fit_indices, fit_weights = GRAD._fit_row_weights(CN, battery, fit_tags)
    gradient_rows = CN.rows().cpu()[fit_indices, :257].long().contiguous()
    plain = GRAD._fit_plain_pca(C.m, generic_rows, manual_logits)
    with torch.enable_grad():
        gradients = {layer: GRAD._fit_loss_gradients(
            C.m, gradient_rows, fit_weights, layer, manual_logits
        ) for layer in GRAD.LAYERS}

    variants: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {
        "mlp_pca_p0_8_r256": {0: pca256[0], 8: pca256[8]},
        "mlp_pca_p0_17_r256": {0: pca256[0], 17: pca256[17]},
        "mlp_pca_p8_17_r256": {8: pca256[8], 17: pca256[17]},
        "mlp_pca_p8_17_r384": {
            layer: (basis[:, :384], mean) for layer, (basis, mean) in rankfit.items()},
        "mlp_pca_p8_17_r512": {
            layer: (basis[:, :512], mean) for layer, (basis, mean) in rankfit.items()},
    }
    hybrid_diagnostics: dict[str, Any] = {"gradient_fit_rows": fit_indices.tolist()}
    for dimension in (32, 64):
        name = f"mlp_pca_grad{dimension}"
        variants[name] = {}; hybrid_diagnostics[name] = {}
        for layer in GRAD.LAYERS:
            basis, mean = plain[layer]
            hybrid, overlap, capture = GRAD._hybrid_basis(basis, gradients[layer], dimension)
            variants[name][layer] = (hybrid, mean)
            hybrid_diagnostics[name][str(layer)] = {
                "overlap_with_pca256": overlap, "fit_gradient_energy_captured": capture}
    if tuple(variants) != IDS:
        raise RuntimeError("MLP-PCA candidate IDs changed")
    return variants, hybrid_diagnostics


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

    C.CROWS, C.CBASE, C.NFLAT = rows, torch.zeros(96 * 256), 96 * 256
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
        raise RuntimeError("complete MLP-PCA candidate identity changed")
    if condition in ("knockout", "partner") and intervention["calls"] != len(IDS) * 24:
        raise RuntimeError(f"{condition} call count changed: {intervention['calls']}")
    payload = {
        "schema": "simplicity_mlp_pca_complete_condition_v1", "condition": condition,
        "candidate_ce": {name: cevs[name].float().cpu() for name in IDS},
        "observed": observed, "qk_indices": index_sets, "qk_widths": sorted(widths),
        "active_replacements": active, "intervention_calls": intervention["calls"],
        "mean_n": mean_n, "identity": identity, "fit_diagnostics": diagnostics,
        "run_bridge": {"L2CF": C.SEL.get("L2CF"), "runtime_s": run.get("runtime_s")},
    }
    torch.save(payload, CONDITION_FILES[condition])
    print(f"RUNG450 child {condition} saved {CONDITION_FILES[condition].name}", flush=True)


def parent() -> None:
    started = time.time()
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    existing = [path.exists() for path in CONDITION_FILES.values()]
    if OUT.exists() or BUNDLE.exists() or any(existing):
        raise RuntimeError("rung450 output namespace exists; preserve and reregister before rerun")
    bank = {row["candidate_id"]: row for row in json.loads(BANK.read_text())["rows"]}
    if any(bank[name]["price_scalars"] != PRICES[name] or bank[name]["family_role"] != "teaching"
           for name in IDS):
        raise RuntimeError("MLP-PCA bank prices/roles changed")
    source = ROOT / "ops/simplicity_mlp_pca_complete_candidate_consequences.py"
    for condition in CONDITIONS:
        environment = dict(os.environ); environment.pop("BQLIB_DRYRUN", None)
        environment["RUNG450_CHILD"] = condition
        subprocess.run([sys.executable, str(source)], env=environment, check=True)

    data = {name: torch.load(path, map_location="cpu", weights_only=True)
            for name, path in CONDITION_FILES.items()}
    native_payload = torch.load(NATIVE, map_location="cpu", weights_only=True)["native_aux"]
    native = native_payload["native_ce"].float(); native_ko = native_payload["native_ko_ce"].float()
    partner = native_payload["partner_ce"].float()
    all_index = torch.arange(native.numel())
    waves = [torch.arange(0, 48 * 256), torch.arange(48 * 256, 96 * 256)]
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

    removal = [arms[name]["full"]["removal_normalized_error"] for name in IDS]
    composition = [arms[name]["full"]["composition_normalized_error"] for name in IDS]
    removal_span = max(removal) - min(removal); composition_span = max(composition) - min(composition)
    ladder = ("mlp_pca_p8_17_r256", "mlp_pca_p8_17_r384", "mlp_pca_p8_17_r512")
    ladder_removal = [arms[name]["full"]["removal_normalized_error"] for name in ladder]
    ladder_composition = [arms[name]["full"]["composition_normalized_error"] for name in ladder]
    rank_removal = spearman([256, 384, 512], [-value for value in ladder_removal])
    rank_composition = spearman([256, 384, 512], [-value for value in ladder_composition])
    removal_wave = spearman(
        [arms[name]["waves"][0]["removal_normalized_error"] for name in IDS],
        [arms[name]["waves"][1]["removal_normalized_error"] for name in IDS])
    composition_wave = spearman(
        [arms[name]["waves"][0]["composition_normalized_error"] for name in IDS],
        [arms[name]["waves"][1]["composition_normalized_error"] for name in IDS])
    identity = all(item["identity"] for item in data.values()) and all(
        data[name]["observed"] == data["unablated"]["observed"]
        and data[name]["active_replacements"] == data["unablated"]["active_replacements"]
        for name in CONDITIONS)
    live = bool(data["knockout"]["intervention_calls"] == len(IDS) * 24
                and data["partner"]["intervention_calls"] == len(IDS) * 24
                and all(data[name]["mean_n"] == 128 * 256 for name in CONDITIONS))
    pred_a = bool(identity and live)
    pred_b = bool(rank_removal >= .999999 and rank_composition >= .999999)
    pred_c = bool(removal_span >= .015 and composition_span >= .015)
    pred_d = bool(removal_wave >= .70 and composition_wave >= .70)
    candidate_live = all(abs(arms[name]["full"]["candidate_ce_damage"]) >= 1e-4 for name in IDS)
    native_effect = native_ko - native
    strong_null = bool(not pred_a or native_effect.norm() <= 1e-6
                       or abs(float((partner - native).mean())) < 1e-4 or not candidate_live
                       or (removal_span < .003 and composition_span < .003)
                       or removal_wave < 0 or composition_wave < 0)
    bundle = {"schema": "simplicity_mlp_pca_complete_candidate_bundle_v1",
              "native_ce": native.half(), "native_ko_ce": native_ko.half(),
              "partner_ce": partner.half(), "arms": bundle_arms}
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 450,
        "claim_level": "prospective_complete_candidate_teaching_label_generation",
        "sealed_opened": False, "condition_files": {
            name: {"path": str(path), "sha256": sha256(path)} for name, path in CONDITION_FILES.items()},
        "complete_identity_all_conditions": identity,
        "intervention_counts": {name: data[name]["intervention_calls"] for name in CONDITIONS},
        "arms": arms, "removal_error_span": removal_span, "composition_error_span": composition_span,
        "rank_ladder": list(ladder), "rank_removal_spearman": rank_removal,
        "rank_composition_spearman": rank_composition,
        "removal_wave_order_spearman": removal_wave,
        "composition_wave_order_spearman": composition_wave,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        'pred_a_complete_instrument': pred_a,
        'pred_b_fixed_rank_ladder_orders_both': pred_b,
        'pred_c_labels_vary': pred_c,
        'pred_d_wave_stability': pred_d,
        "strong_null_complete_family_unusable": strong_null,
        "next_step": ("count_mlp_pca_family_and_build_vocabulary_teaching_family"
                      if pred_a and pred_b and pred_c and pred_d and not strong_null
                      else "do_not_count_mlp_pca_family"),
        "new_deployed_values": 0, "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() and sha256(path) == digest for path, digest in HASHES.items())
        assert len(IDS) == 7 and set(IDS) == set(PRICES)
        print("RUNG450 COMPLETE MLP-PCA CONSEQUENCES | dry run: hashes, candidates, bars valid")
        return
    mode = os.environ.get("RUNG450_CHILD")
    if mode:
        child(mode)
    else:
        parent()


if __name__ == "__main__":
    main()
