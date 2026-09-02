"""RUNG456 -- INDEPENDENT VOCABULARY FIXED-SCALE REPRODUCTION."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from receipt import dump
import simplicity_vocabulary_complete_candidate_consequences as V


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
OUT = ROOT / "simplicity_vocabulary_fixed_scale_independent_results.json"
BUNDLE = ROOT / "simplicity_vocabulary_fixed_scale_independent_bundle.pt"
ROWS = ROOT / ".rowcache_simplicity_mlp_pca_reliability_v1/reliability_192.pt"
ROWS_RECEIPT = ROOT / "simplicity_mlp_pca_reliability_v1_rows_receipt.json"
SPEC = ROOT / "simplicity_fixed_scale_composition_spec_v1.json"
OLD_RESULT = ROOT / "simplicity_vocabulary_complete_candidate_consequences_results.json"
BANK = POLY / "prospective_consequence_candidate_bank_v1.json"
FIT480 = ROOT / ".rowcache/fineweb_n480_skip80.pt"
MEAN_FIT = ROOT / ".rowcache/fineweb_n192_skip11000.pt"
FACTOR = ROOT / "mlp16_rank2_quadratic_factored.pt"
DOCUMENTS = 192
TOKENS = 256
D = 1152
W = 50304
IDS = V.IDS
PRICES = V.PRICES
HASHES = {
    BANK: "e35d5c0aa1dae34173b93ae4d81cafa8317539adfaf7c74bfe7decb068ac47be",
    ROWS_RECEIPT: "d991469476d9f2623f011cfdb57a768fe41bb9589e24df67ccc095497c21ebc3",
    ROWS: "c340a75c27afa0c37f4404c8ddc865275ed626542260c7f84ae14dba7c1ccbbc",
    SPEC: "c40cea4a846dec83eddacf7623caa49b82e351e4f94be4dd9caf86da9e8467b2",
    OLD_RESULT: "2275efc6f2f690e30f8e212c7b3b2d32e37e15c031c94d201372a74b0557e27d",
    FIT480: "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496",
    MEAN_FIT: "b1564bfd071418f401a816cb01e3d26b082a3e73ba858838f1c83c250db4d868",
    FACTOR: "b9870f738b528e988ff9a1e04cdc6e1096de8ab0dc5fa86bb76229812d9ffb6e",
    ROOT / "ops/simplicity_vocabulary_complete_candidate_consequences.py":
        "3ab9150671494036ea09a124c8a0f212547cbbc03b7cbba0f49410bf52600b3c",
    ROOT / "ops/joint_vocab_distributed_rank_frontier.py":
        "76a6525b67e6128d733c62343741d79f058072966b38a06af4ad2add6c4421de",
    ROOT / "ops/joint_vocab_sparse_rare_residual.py":
        "18c38fdbc6849dd4aa33ff30b58471a02c0427dfcf6bf16d842b6cb9c9a83532",
}


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fixed_score(native: torch.Tensor, partner: torch.Tensor, candidate: torch.Tensor,
                joint: torch.Tensor, index: torch.Tensor) -> dict[str, float]:
    interaction = ((joint - native).double()
                   - ((candidate - native) + (partner - native)).double())[index]
    partner_effect = (partner - native).double()[index]
    numerator = float(interaction.norm())
    denominator = float(partner_effect.norm())
    return {
        "interaction_l2": numerator,
        "partner_effect_l2": denominator,
        "fixed_scale_composition_error": numerator / max(denominator, 1e-12),
        "interaction_rms": float(interaction.square().mean().sqrt()),
    }


def mean_absolute_difference(left: list[float], right: list[float]) -> float:
    return float(torch.tensor(left, dtype=torch.float64).sub(
        torch.tensor(right, dtype=torch.float64)).abs().mean())


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() and sha256(path) == digest for path, digest in HASHES.items())
        assert len(IDS) == 23 and len(V.rank_edges()) == 14
        print("RUNG456 INDEPENDENT VOCABULARY FIXED SCALE | dry run: hashes,23 candidates,14 edges valid")
        return
    started = time.time()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung456 output namespace already exists")
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    bank = {row["candidate_id"]: row for row in json.loads(BANK.read_text())["rows"]}
    if any(bank[name]["price_scalars"] != PRICES[name]
           or bank[name]["program_family"] != "vocabulary_factorization"
           or bank[name]["family_role"] != "teaching" for name in IDS):
        raise RuntimeError("vocabulary bank identity/price/role changed")
    spec = json.loads(SPEC.read_text())
    old_fixed = {name: arm["fixed_scale_composition_error"]
                 for name, arm in spec["families"]["vocabulary"]["arms"].items()}
    old_result = json.loads(OLD_RESULT.read_text())
    if tuple(old_fixed) != IDS or tuple(old_result["arms"]) != IDS:
        raise RuntimeError("old vocabulary arm identity changed")

    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    from bilin18_observed_model_facade import validate_production_model
    import joint_vocab_distributed_rank_frontier as distributed
    import joint_vocab_sparse_rare_residual as rare
    from tier2_model import load_elriggs

    model, config = load_elriggs("bilin18")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    validate_production_model(model)
    embedding = model.transformer.wte.weight.detach().float().contiguous()
    output = model.lm_head.weight.detach().float().contiguous()
    if tuple(embedding.shape) != (W, D) or tuple(output.shape) != (W, D) or config["n_embd"] != D:
        raise RuntimeError("native vocabulary matrix identity changed")

    V.DOCUMENTS = DOCUMENTS
    rows = V.load_rows(ROWS, DOCUMENTS)
    if tuple(rows.shape) != (DOCUMENTS, TOKENS + 1):
        raise RuntimeError("independent row geometry changed")
    target = rows[:, 1:].reshape(-1).long().contiguous()
    calibration = V.load_rows(MEAN_FIT, 128)
    mean_value, mean_n = V.attention_mean(model, calibration)
    factor_cpu = torch.load(FACTOR, map_location="cpu", weights_only=True)
    factor = {key: value.cuda().float() for key, value in factor_cpu.items()}
    native_hidden, native_calls = V.capture_hidden(model, rows, "native", None, None)
    replay_hidden, replay_calls = V.capture_hidden(model, rows, "native", None, None)
    knockout_hidden, knockout_calls = V.capture_hidden(model, rows, "knockout", mean_value, None)
    partner_hidden, partner_calls = V.capture_hidden(model, rows, "partner", None, factor)
    hidden_replay_max = float((replay_hidden - native_hidden).abs().max())
    if native_calls or replay_calls or knockout_calls != 48 or partner_calls != 48:
        raise RuntimeError("independent consequence capture counts changed")

    uniform = V.fit_uniform(embedding, output)
    uniform["residual_code"] = uniform["residual"] @ uniform["residual_vectors"][:, :512]
    uniform["output_code"] = output @ uniform["output_vectors"][:, :537]
    fit = V.load_rows(FIT480, 480)
    counts = torch.bincount(fit[:, 1:].reshape(-1), minlength=W).float()
    weights = {
        "sqrt_count_plus_1": (counts.cuda() + 1.0).sqrt(),
        "count_plus_1": counts.cuda() + 1.0,
    }
    weighted = {}
    for name, weight in weights.items():
        weight = weight / weight.mean()
        fitted = distributed._fit_full(embedding, output, weight)
        fitted["residual_code"] = fitted["residual"] @ fitted["residual_vectors"][:, :768]
        fitted["output_code"] = output @ fitted["output_vectors"][:, :793]
        weighted[name] = fitted

    count_fit = weighted["count_plus_1"]
    count_basis = count_fit["residual_vectors"][:, :512]
    count_code = count_fit["residual"] @ count_basis
    approximation = embedding @ count_fit["mapping"] + count_code @ count_basis.T
    correction = output - approximation
    rare_allowed = torch.zeros(W, dtype=torch.bool)
    rare_allowed[:V.V_REAL] = counts[:V.V_REAL].cpu() <= 2
    selection_hidden, _ = V.capture_hidden(model, fit[:32], "native", None, None)

    def shared_function(hidden: torch.Tensor) -> torch.Tensor:
        return ((hidden @ count_fit["mapping"].T) @ embedding.T
                + (hidden @ count_basis) @ count_code.T)

    fisher_score = rare._fisher_scores(selection_hidden, output, shared_function)
    norm_score = correction.square().sum(1).double().cpu()
    fisher_ids_cpu = rare._top_allowed(fisher_score, rare_allowed, V.SPARSE_K)
    norm_ids_cpu = rare._top_allowed(norm_score, rare_allowed, V.SPARSE_K)
    rare_ids = torch.nonzero(rare_allowed, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(V.RANDOM_SEED)
    random_ids_cpu = rare_ids[torch.randperm(len(rare_ids), generator=generator)[:V.SPARSE_K]]
    selections = {"fisher": fisher_ids_cpu, "norm": norm_ids_cpu, "random": random_ids_cpu}
    old_selection = old_result["sparse_selection_receipts"]
    for name, ids in selections.items():
        if V.tensor_sha256(ids) != old_selection[name]["semantic_sha256"]:
            raise RuntimeError(f"sparse selection changed: {name}")
    sparse = {}
    for name, ids_cpu in selections.items():
        ids = ids_cpu.cuda()
        sparse[f"{name}_ids"] = ids
        sparse[f"{name}_rows"] = correction[ids].contiguous()

    native = V.score_native(native_hidden, target, output)
    native_replay = V.score_native(replay_hidden, target, output)
    native_ko = V.score_native(knockout_hidden, target, output)
    partner = V.score_native(partner_hidden, target, output)
    condition_scores = {
        "unablated": V.score_all(native_hidden, target, embedding, uniform, weighted, sparse),
        "knockout": V.score_all(knockout_hidden, target, embedding, uniform, weighted, sparse),
        "partner": V.score_all(partner_hidden, target, embedding, uniform, weighted, sparse),
    }

    all_index = torch.arange(native.numel())
    waves = [torch.arange(0, 96 * TOKENS), torch.arange(96 * TOKENS, 192 * TOKENS)]
    arms = {}
    bundle_arms = {}
    for name in IDS:
        candidate = condition_scores["unablated"][name]
        candidate_ko = condition_scores["knockout"][name]
        joint = condition_scores["partner"][name]
        removal_full = V.metrics(native, native_ko, candidate, candidate_ko,
                                 partner, joint, all_index)["removal_normalized_error"]
        fixed_full = fixed_score(native, partner, candidate, joint, all_index)
        wave_rows = []
        for index in waves:
            removal = V.metrics(native, native_ko, candidate, candidate_ko,
                                partner, joint, index)["removal_normalized_error"]
            wave_rows.append({"removal_normalized_error": removal,
                              **fixed_score(native, partner, candidate, joint, index)})
        arms[name] = {
            "price_scalars": PRICES[name],
            "full": {"removal_normalized_error": removal_full, **fixed_full},
            "waves": wave_rows,
        }
        bundle_arms[name] = {
            "candidate_ce": candidate.half(),
            "candidate_ko_ce": candidate_ko.half(),
            "candidate_partner_ce": joint.half(),
        }

    fixed_values = {name: arms[name]["full"]["fixed_scale_composition_error"] for name in IDS}
    removal_values = {name: arms[name]["full"]["removal_normalized_error"] for name in IDS}
    old_list = [old_fixed[name] for name in IDS]
    new_list = [fixed_values[name] for name in IDS]
    old_new_pearson = V.pearson(old_list, new_list)
    old_new_mae = mean_absolute_difference(old_list, new_list)
    structured = {
        "fixed_composition": V.edge_accuracy(fixed_values),
        "removal": V.edge_accuracy(removal_values),
    }
    wave_values = [{
        "fixed_composition": {name: arms[name]["waves"][wave]["fixed_scale_composition_error"] for name in IDS},
        "removal": {name: arms[name]["waves"][wave]["removal_normalized_error"] for name in IDS},
    } for wave in range(2)]
    wave_structured = [{metric: V.edge_accuracy(wave_values[wave][metric])
                        for metric in ("fixed_composition", "removal")} for wave in range(2)]
    wave_pearson = {metric: V.pearson(
        [wave_values[0][metric][name] for name in IDS],
        [wave_values[1][metric][name] for name in IDS])
        for metric in ("fixed_composition", "removal")}

    replay_ce_max = float((native_replay - native).abs().max())
    # pred_a complete instrument preserves exact replay, mean positions, intervention calls, and identities.
    pred_a = bool(hidden_replay_max == 0 and replay_ce_max == 0 and mean_n == 128 * TOKENS
                  and knockout_calls == partner_calls == 48)
    # pred_b independent continuous reproduction meets old-to-new correlation and mean-shift bars.
    pred_b = bool(old_new_pearson >= .90 and old_new_mae <= .10)
    # pred_c independent structured order preserves fixed composition and removal rank edges.
    pred_c = bool(all(item["accuracy"] >= .85 for item in structured.values()))
    # pred_d wave reliability preserves rank edges and continuous values in both document waves.
    pred_d = bool(all(wave_structured[wave][metric]["accuracy"] >= .70
                      for wave in range(2) for metric in ("fixed_composition", "removal"))
                  and all(value >= .70 for value in wave_pearson.values()))
    partner_live = float((partner - native).double().norm()) > 1e-6
    candidates_live = all(abs(float((condition_scores["unablated"][name] - native).mean())) >= 1e-4
                          for name in IDS)
    strong_null = bool(not pred_a or not partner_live or not candidates_live or old_new_pearson < 0
                       or any(item["accuracy"] < .50 for item in structured.values())
                       or any(value < 0 for value in wave_pearson.values()))
    if not all(math.isfinite(value) for value in (old_new_pearson, old_new_mae, *wave_pearson.values())):
        raise RuntimeError("non-finite independent statistic")

    bundle = {
        "schema": "simplicity_vocabulary_fixed_scale_independent_bundle_v1",
        "native_ce": native.half(), "native_ko_ce": native_ko.half(),
        "partner_ce": partner.half(), "arms": bundle_arms,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 456,
        "claim_level": "independent_fixed_scale_metric_reproduction_not_family_salvage",
        "sealed_opened": False, "candidate_ids": list(IDS), "candidate_prices": PRICES,
        "program_scope": "exact_input_embedding_candidate_output_vocabulary_map",
        "native_hidden_replay_max_abs": hidden_replay_max,
        "native_ce_replay_max_abs": replay_ce_max,
        "capture_counts": {"attention_mean_positions": mean_n,
                           "knockout_calls": knockout_calls, "partner_calls": partner_calls},
        "sparse_selection_semantic_sha256": {
            name: V.tensor_sha256(ids) for name, ids in selections.items()},
        "arms": arms,
        "old_to_independent_fixed_composition": {
            "pearson": old_new_pearson, "mean_absolute_shift": old_new_mae,
        },
        "structured_rank_accuracy": structured,
        "wave_structured_rank_accuracy": wave_structured,
        "wave_continuous_pearson": wave_pearson,
        "bundle": {"path": str(BUNDLE), "sha256": sha256(BUNDLE)},
        'pred_a_complete_instrument': pred_a,
        'pred_b_independent_continuous_reproduction': pred_b,
        'pred_c_independent_structured_order': pred_c,
        'pred_d_wave_reliability': pred_d,
        "strong_null_fixed_scale_not_reproduced": strong_null,
        "scientific_status": {
            "rung454_repaired": False, "vocabulary_family_counted": False,
            "teaching_family_count": 2, "predictor_fit": False,
        },
        "runtime_s": time.time() - started,
        "next_step": ("freeze_protocol_v2_decision_and_find_new_third_teaching_family"
                      if pred_a and pred_b and pred_c and pred_d and not strong_null
                      else "preserve_fixed_scale_null_and_find_new_third_teaching_family"),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 456,
        "predictions": {"A": pred_a, "B": pred_b, "C": pred_c, "D": pred_d},
        "strong_null": strong_null,
        "old_to_new": result["old_to_independent_fixed_composition"],
        "structured": {key: value["accuracy"] for key, value in structured.items()},
        "wave_structured": [{key: value["accuracy"] for key, value in row.items()}
                            for row in wave_structured],
        "wave_pearson": wave_pearson,
        "bundle_sha256": result["bundle"]["sha256"],
        "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
