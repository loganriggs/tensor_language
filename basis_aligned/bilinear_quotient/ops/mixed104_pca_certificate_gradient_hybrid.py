"""RUNG 313 -- SAME-RANK CERTIFICATE-GRADIENT HYBRID FOR MLPs 8 AND 17.

The fixed-pair rank frontier follows an almost one-dimensional certificate
damage law.  Ordinary PCA only reduces its magnitude.  Test a qualitatively
different rank-256 basis at the identical {8,17} price by reserving k output
directions for native certificate-loss gradients:

    pca256: 256 activation-PCA directions
    grad32: 224 activation-PCA + 32 orthogonal gradient directions
    grad64: 192 activation-PCA + 64 orthogonal gradient directions

The gradient measure uses the lexicographically even half of the 62 tags.
Exactly 16 census rows are selected deterministically by weighted fit-tag
coverage.  Those rows are excluded from every reported certificate score; the
lexicographically odd tag half is the decisive transfer set.  PCA still uses
the old 16 FineWeb fit rows.  Every arm saves 7,667,712 scalars and proposes
531,927,350 standalone scalars.

For a projector P=QQ^T, first-order loss obeys

    |g^T(I-P)(y-mu)| <= ||(I-P)g|| ||(I-P)(y-mu)||.

PCA optimizes only the second factor; the hybrids spend rank on the first.

Frozen predictions
------------------
pred_a_hybrid_improves_full_certificates_at_equal_price:
    Some hybrid gains >=8 row-excluded full-battery certificates over pca256,
    has >=27/62 valid, and costs at most +.01 extra census damage.
pred_b_improvement_transfers_to_heldout_tags:
    The same hybrid gains >=4 valid certificates on the 31 heldout tags and
    lowers their median normalized member damage by >=10%.
pred_c_hybrid_breaks_the_one_dimensional_damage_law:
    The historical per-tag affine threshold model predicts pca256 within 3
    certificates, while the qualifying hybrid beats its own prediction by >=5.

Null: neither hybrid gains two full-battery certificates, or both add >=.01
census damage.  This is a split-tag, removed-row physical falsifier; fresh/OOD
and causal confirmation remain necessary after a pass.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
OUT = ROOT / "mixed104_pca_certificate_gradient_hybrid_results.json"
LAYERS = (8, 17)
RANK = 256
GRAD_DIMS = (32, 64)
FIT_ROWS = 16
D = 1152
H = 4608
ADOPTED_SCALARS = 539_595_062
ADOPTED_BYTES = 2_042_438_252
SAVING = 2 * (H * D - RANK * (H + D))


def _threshold(receipt: dict[str, object]) -> float:
    return 0.5 * float(receipt["mean_ablation"]["top"][0]["abs_dce_members"])


def _certificate_stats(CN, battery, damage, tags, excluded_rows):
    valid = 0
    member_abs = {}
    normalized = []
    excluded_rows = set(int(value) for value in excluded_rows)
    for tag in tags:
        receipt = battery[tag]
        member = CN.leaf(tag)["member"].long()
        keep = torch.tensor([int(index) // 256 not in excluded_rows for index in member], dtype=torch.bool)
        member = member[keep]
        if member.numel() == 0:
            raise RuntimeError(f"all members excluded for {tag}")
        value = float(damage[member].abs().mean())
        threshold = _threshold(receipt)
        member_abs[tag] = round(value, 7)
        normalized.append(value / threshold)
        valid += int(value < threshold)
    return valid, member_abs, float(torch.tensor(normalized).median())


def _fit_row_weights(CN, battery, fit_tags):
    weights = torch.zeros(CN.nflat(), dtype=torch.float64)
    for tag in fit_tags:
        member = CN.leaf(tag)["member"].long()
        # Equal total mass per tag, with inverse-margin emphasis.
        weights[member] += 1.0 / (_threshold(battery[tag]) * member.numel())
    matrix = weights.reshape(-1, 256)
    row_score = matrix.sum(1)
    selected = torch.argsort(row_score, descending=True, stable=True)[:FIT_ROWS]
    chosen = matrix[selected]
    positive_mean = chosen[chosen > 0].mean()
    chosen = chosen / positive_mean
    assert bool(torch.isfinite(chosen).all()) and float(chosen.sum()) > 0
    return selected.long(), chosen.float()


@torch.no_grad()
def _fit_plain_pca(model, rows, manual_logits):
    captured = {layer: [] for layer in LAYERS}
    handles = []
    for layer in LAYERS:
        def hook(_module, _args, output, layer=layer):
            captured[layer].append(output.detach().float().cpu().reshape(-1, D))
        handles.append(model.transformer.h[layer].mlp.register_forward_hook(hook))
    try:
        for start in range(0, len(rows), 2):
            manual_logits(model, rows[start:start + 2, :-1].cuda())
    finally:
        for handle in handles:
            handle.remove()

    fitted = {}
    for layer in LAYERS:
        output = torch.cat(captured[layer]).cuda()
        mean = output.mean(0)
        centered = output - mean
        covariance = centered.T @ centered / len(centered)
        covariance = 0.5 * (covariance + covariance.T)
        jitter = 1e-7 * float(torch.diagonal(covariance).mean().abs().clamp_min(1e-12))
        values, vectors = torch.linalg.eigh(covariance + jitter * torch.eye(D, device="cuda"))
        order = torch.argsort(values, descending=True)[:RANK]
        fitted[layer] = (vectors[:, order].cpu(), mean.cpu())
        print(f"plain PCA layer {layer}: top256 energy "
              f"{float(values[order].sum()/values.clamp_min(0).sum()):.4f}", flush=True)
    return fitted


def _fit_loss_gradients(model, rows, position_weights, layer, manual_logits):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gradients = []
    capture = {}

    def leaf_hook(_module, _args, output):
        leaf = output.detach().float().requires_grad_(True)
        capture["leaf"] = leaf
        return leaf.to(output.dtype)

    handle = model.transformer.h[layer].mlp.register_forward_hook(leaf_hook)
    try:
        for start in range(0, len(rows), 2):
            batch = rows[start:start + 2]
            index, target = batch[:, :-1].cuda(), batch[:, 1:].cuda()
            logits = manual_logits(model, index)
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), target.reshape(-1), reduction="none"
            ).reshape(len(batch), 256)
            loss = (per_token * position_weights[start:start + len(batch)].cuda()).sum()
            loss.backward()
            leaf = capture.pop("leaf")
            if leaf.grad is None:
                raise RuntimeError(f"missing layer-{layer} gradient")
            gradients.append(leaf.grad.detach().float().cpu().reshape(-1, D))
    finally:
        handle.remove()
    gradient = torch.cat(gradients)
    assert gradient.shape == (len(rows) * 256, D) and bool(torch.isfinite(gradient).all())
    print(f"gradient layer {layer}: rms {float(gradient.square().mean().sqrt()):.6g}", flush=True)
    return gradient


def _hybrid_basis(pca_basis, gradient, gradient_dim):
    q0 = pca_basis[:, : RANK - gradient_dim].cuda()
    grad = gradient.cuda()
    scale = grad.square().mean().sqrt().clamp_min(1e-20)
    grad = grad / scale
    residual = grad - (grad @ q0) @ q0.T
    covariance = residual.T @ residual / len(residual)
    covariance = 0.5 * (covariance + covariance.T)
    jitter = 1e-7 * float(torch.diagonal(covariance).mean().abs().clamp_min(1e-12))
    values, vectors = torch.linalg.eigh(covariance + jitter * torch.eye(D, device="cuda"))
    grad_basis = vectors[:, torch.argsort(values, descending=True)[:gradient_dim]]
    hybrid = torch.linalg.qr(torch.cat((q0, grad_basis), dim=1), mode="reduced").Q
    overlap = float((pca_basis.cuda().T @ hybrid).square().sum() / RANK)
    capture = float((grad @ hybrid).square().sum() / grad.square().sum())
    return hybrid.cpu(), overlap, capture


def _historical_affine_prediction(battery, tags, census_damage):
    configs = []
    for filename in ("mixed104_online_cv0_results.json",
                     "mixed104_online_cv0_pca_fixed_triple_results.json"):
        receipt = json.loads((ROOT / filename).read_text())
        configs.append((receipt["census_damage"], receipt["member_abs_dce"]))
    pairs = json.loads((ROOT / "mixed104_pca_fixed_pair_frontier_results.json").read_text())
    for arm in pairs["arms"].values():
        configs.append((arm["census_damage"], arm["member_abs_dce"]))
    x = torch.tensor([[1.0, value] for value, _ in configs], dtype=torch.float64)
    y = torch.tensor([[members[tag] for tag in tags] for _, members in configs], dtype=torch.float64)
    coefficients = torch.linalg.lstsq(x, y).solution
    predicted = torch.tensor([1.0, census_damage], dtype=torch.float64) @ coefficients
    return sum(float(predicted[index]) < _threshold(battery[tag]) for index, tag in enumerate(tags))


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for filename in ("mixed104_pca_fixed_pair_rank_frontier_results.json",
                         "mixed104_pca_fixed_pair_frontier_results.json",
                         "mixed104_online_cv0_pca_fixed_triple_results.json",
                         "mixed104_online_cv0_results.json", "census_state_diverse.pt"):
            assert (ROOT / filename).exists(), filename
        assert SAVING == 7_667_712
        assert ADOPTED_SCALARS - SAVING == 531_927_350
        assert ADOPTED_BYTES - 4 * SAVING == 2_011_767_404
        print("MIXED104 PCA CERTIFICATE-GRADIENT HYBRID | dry run: split, price, variants, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "ops"))
    sys.path.insert(0, str(ROOT.parent / "qk_mdl"))
    sys.path.insert(0, "/workspace/rspd")
    import census_lib as CN
    import cevdump_ct96 as C
    import mlp_activation_pca_four_layer_composition as pca_base
    from mlp0_signed_response_rank_screen import _manual_logits

    CN.use_state("census_state_diverse.pt")
    rows, base_ce, nflat = CN.rows().cpu(), CN.base_ce().float().cpu(), CN.nflat()
    C.CROWS, C.CBASE, C.NFLAT = rows, base_ce, nflat
    C.ANCH = json.loads((ROOT / "frontier_tail_traj_results.json").read_text())
    battery = json.loads((ROOT / "circuits/BATTERY.json").read_text())["by_tag"]
    tags = sorted(tag for tag in battery if tag in set(CN.all_tags()))
    assert len(tags) == 62
    fit_tags, heldout_tags = tags[::2], tags[1::2]
    fit_row_indices, fit_position_weights = _fit_row_weights(CN, battery, fit_tags)
    gradient_rows = rows[fit_row_indices, :257].long().contiguous()
    print(f"gradient fit rows: {fit_row_indices.tolist()}", flush=True)

    generic_rows = pca_base._load_rows(ROOT / ".rowcache/fineweb_n480_skip80.pt", FIT_ROWS)
    plain = _fit_plain_pca(C.m, generic_rows, _manual_logits)
    gradients = {layer: _fit_loss_gradients(
        C.m, gradient_rows, fit_position_weights, layer, _manual_logits
    ) for layer in LAYERS}

    variants = {"pca256": plain}
    diagnostics = {}
    for gradient_dim in GRAD_DIMS:
        projectors = {}
        diagnostics[f"grad{gradient_dim}"] = {}
        for layer in LAYERS:
            basis, mean = plain[layer]
            hybrid, overlap, capture = _hybrid_basis(basis, gradients[layer], gradient_dim)
            projectors[layer] = (hybrid, mean)
            diagnostics[f"grad{gradient_dim}"][str(layer)] = {
                "overlap_with_pca256": overlap,
                "fit_gradient_energy_captured": capture,
            }
        variants[f"grad{gradient_dim}"] = projectors

    C.SEL.update({
        "mode": "norm", "K": 4608, "K69": 4608, "K69MAP": {},
        "skipset": tuple(range(10, 18)), "motif_off": (), "clsdmg": True,
        "ext_rows": rows, "cp_swap": 4608, "qk_r": 96, "qk_rmap": {},
        "qk_extra_tail": 8, "qk_tail": True, "drop_tailE": True,
        "drop_a1v": True, "drop_a0": True,
        "final_mlp_projectors": variants["pca256"],
        "final_mlp_projector_variants": variants,
        "final_mlp_primary_variant": "pca256",
    })
    print("ARM FAMILY: mixed104 + {8,17}@r256 PCA/grad32/grad64", flush=True)
    run = C.main()
    variant_cevs = C.SEL.get("_final_mlp_variant_cevs", {})
    observed = C.SEL.get("_final_mlp_variant_observed", {})
    if set(variant_cevs) != set(variants) or set(observed) != set(variants):
        raise SystemExit("INSTRUMENT FAIL: missing hybrid variant output")
    for name in variants:
        got = {int(key): int(value) for key, value in observed[name].items()}
        if got != {8: RANK, 17: RANK}:
            raise SystemExit(f"INSTRUMENT FAIL: {name} observed {got}")

    wanted = tuple(list(range(96)) + list(range(120, 128)))
    index_sets = C.SEL.get("_QK_INDEX_SETS", {})
    qk = C.SEL.get("_QKR", {})
    widths = {int(factor[0].shape[1]) for heads in qk.values()
              for factors in heads.values() for factor in factors}
    active = tuple(C.SEL.get("_ORDER2", ()))
    if (set(index_sets) != set(range(2, 18)) or any(value != wanted for value in index_sets.values())
            or widths != {104} or any(name in active for name in ("a0", "a1v", "tailE"))):
        raise SystemExit("INSTRUMENT FAIL: mixed104 identity changed")

    arms = {}
    excluded = fit_row_indices.tolist()
    for name, cev in variant_cevs.items():
        damage_vector = cev.float().reshape(-1) - base_ce
        census_damage = float(damage_vector.mean())
        full_valid, full_abs, full_median = _certificate_stats(
            CN, battery, damage_vector, tags, excluded
        )
        held_valid, held_abs, held_median = _certificate_stats(
            CN, battery, damage_vector, heldout_tags, excluded
        )
        predicted = _historical_affine_prediction(battery, tags, census_damage)
        arms[name] = {
            "census_damage": census_damage,
            "row_excluded_full_certificates_valid": full_valid,
            "row_excluded_heldout_tag_certificates_valid": held_valid,
            "row_excluded_full_median_normalized_damage": full_median,
            "row_excluded_heldout_median_normalized_damage": held_median,
            "historical_affine_predicted_full_certificates": predicted,
            "full_member_abs_dce": full_abs,
            "heldout_member_abs_dce": held_abs,
        }
        print(f"{name}: census {census_damage:+.6f}, full {full_valid}/62, "
              f"heldout {held_valid}/31, predicted {predicted}", flush=True)

    control = arms["pca256"]
    qualifying = []
    for name in ("grad32", "grad64"):
        arm = arms[name]
        full_gain = arm["row_excluded_full_certificates_valid"] - control[
            "row_excluded_full_certificates_valid"]
        held_gain = arm["row_excluded_heldout_tag_certificates_valid"] - control[
            "row_excluded_heldout_tag_certificates_valid"]
        held_ratio = arm["row_excluded_heldout_median_normalized_damage"] / control[
            "row_excluded_heldout_median_normalized_damage"]
        pa = (full_gain >= 8 and arm["row_excluded_full_certificates_valid"] >= 27
              and arm["census_damage"] <= control["census_damage"] + .01)
        pb = held_gain >= 4 and held_ratio <= .90
        pc = (abs(control["row_excluded_full_certificates_valid"]
                  - control["historical_affine_predicted_full_certificates"]) <= 3
              and arm["row_excluded_full_certificates_valid"]
              - arm["historical_affine_predicted_full_certificates"] >= 5)
        if pa and pb and pc:
            qualifying.append(name)
    pred_a = any(
        arms[name]["row_excluded_full_certificates_valid"]
        - control["row_excluded_full_certificates_valid"] >= 8
        and arms[name]["row_excluded_full_certificates_valid"] >= 27
        and arms[name]["census_damage"] <= control["census_damage"] + .01
        for name in ("grad32", "grad64")
    )
    pred_b = any(
        arms[name]["row_excluded_heldout_tag_certificates_valid"]
        - control["row_excluded_heldout_tag_certificates_valid"] >= 4
        and arms[name]["row_excluded_heldout_median_normalized_damage"]
        <= .90 * control["row_excluded_heldout_median_normalized_damage"]
        for name in ("grad32", "grad64")
    )
    pred_c = (abs(control["row_excluded_full_certificates_valid"]
                  - control["historical_affine_predicted_full_certificates"]) <= 3
              and any(arms[name]["row_excluded_full_certificates_valid"]
                      - arms[name]["historical_affine_predicted_full_certificates"] >= 5
                      for name in ("grad32", "grad64")))
    null = (all(arms[name]["row_excluded_full_certificates_valid"]
                - control["row_excluded_full_certificates_valid"] < 2
                for name in ("grad32", "grad64"))
            or all(arms[name]["census_damage"] >= control["census_damage"] + .01
                   for name in ("grad32", "grad64")))
    result = {
        "status": "mixed104_pca_certificate_gradient_hybrid_complete",
        "rung": 313,
        "claim_level": "split_tag_removed_row_equal_rank_certificate_gradient_falsifier_only",
        "convention": "CE added above native; lower is better",
        "fit_split": {"fit_tags": fit_tags, "heldout_tags": heldout_tags,
                      "excluded_gradient_fit_rows": excluded},
        "price_each": {"saving_scalars": SAVING,
                       "literal_standalone_scalars": ADOPTED_SCALARS - SAVING,
                       "literal_raw_tensor_bytes": ADOPTED_BYTES - 4 * SAVING},
        "basis_diagnostics": diagnostics,
        "arms": arms,
        "jointly_qualifying_hybrids": qualifying,
        "primary_pca_fresh8_diagnostic_only": [float(value) for value in run["fresh8"]],
        "mixed_identity": {"qk_indices": list(wanted), "qk_widths": sorted(widths),
                           "active_replacements": list(active)},
        'pred_a_hybrid_improves_full_certificates_at_equal_price': bool(pred_a),
        'pred_b_improvement_transfers_to_heldout_tags': bool(pred_b),
        'pred_c_hybrid_breaks_the_one_dimensional_damage_law': bool(pred_c),
        "null_no_constraint_orthogonal_gain": bool(null),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"qualifying": qualifying, "predicates": [pred_a, pred_b, pred_c],
                      "null": null, "runtime_s": result["runtime_s"]}, indent=2), flush=True)
    print("MIXED104 PCA CERTIFICATE-GRADIENT HYBRID DONE", flush=True)


if __name__ == "__main__":
    main()
