#!/usr/bin/env python3
"""RUNG533 -- four-way downstream equivalence of equality-score factors."""

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
import torch.nn.functional as F

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import bilin18_observed_model_facade as facade
import equality_factor_companion_causal_equivalence_rung532 as parent
import equality_factor_to_slot_exchangeability_rung533_math as math_contract


RUNG = 533
PAIR_NAME = "L8H3->L8H4"
ROLES = ("final_natural", "ood_code")
ROLE_FILES = {
    role: ROOT / f".rowcache_induction_equality_tensor_final_ood_v2/{role}.pt"
    for role in ROLES
}
DOCUMENTS_PER_ROLE = 192
DOCUMENT_SPLIT = 96
TOKENS = 256
BATCH = 4
BACKGROUNDS = parent.BACKGROUNDS
MAPPINGS = tuple(math_contract.SCALES)
CONTROL_BY_MAPPING = {mapping: f"{mapping}_key_control" for mapping in MAPPINGS}
ARMS = (
    "native", "absent", "product_control",
    *(arm for mapping in MAPPINGS for arm in (mapping, CONTROL_BY_MAPPING[mapping])),
)
TASK_CELLS = ("positive", "matched_negative", "off_target")
BATCHES_PER_ROLE = DOCUMENTS_PER_ROLE // BATCH
FORWARDS_PER_BATCH = 1 + len(BACKGROUNDS) * len(ARMS)
FORWARDS = len(ROLES) * BATCHES_PER_ROLE * FORWARDS_PER_BATCH

PREREG = POLY / "EQUALITY_FACTOR_TO_SLOT_EXCHANGEABILITY_RUNG533_PREREGISTRATION.md"
PARENT_SOURCE = ROOT / "ops/equality_factor_companion_causal_equivalence_rung532.py"
PARENT_RESULT = ROOT / "equality_factor_companion_causal_equivalence_rung532_results.json"
MATH_SOURCE = ROOT / "ops/equality_factor_to_slot_exchangeability_rung533_math.py"
MATH_RESULT = ROOT / "equality_factor_to_slot_exchangeability_rung533_math.json"
ROWS_RECEIPT = ROOT / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
FACADE_SOURCE = POLY / "bilin18_observed_model_facade.py"
INDUCTION_SOURCE = POLY / "circuit_induction_tensor.py"
MODEL_SOURCE = Path("/workspace/tensor_language/jacclust/tt_model.py")
OUT = ROOT / "equality_factor_to_slot_exchangeability_rung533_results.json"
BUNDLE = ROOT / "equality_factor_to_slot_exchangeability_rung533_bundle.pt"
HASHES = {
    PREREG: "d5ed32a7a4268768ed170e4a0fdd282fb49e3be97190c077366e77353a6ad1eb",
    PARENT_SOURCE: "142f4a0f05d582413fb6eac1820654dc6d4491690af9742e0a2d81eac719fdb8",
    PARENT_RESULT: "76b7c417a9bceff2f35937f51404c5248bac19b3024fb32ec6891ae70ae4ba2b",
    MATH_SOURCE: "f0132d5feb91eb149796f123d48e99b75fbdf8da2bede83e1443593d45be5272",
    MATH_RESULT: "191dc257c0c253a9eac942426a97b792a0c74ce405ff12cfc920029447386958",
    ROWS_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    ROLE_FILES["final_natural"]: "5f2813eacc3ec66162c2ce695b978264137c66126fdc25e3d49b4efd44a9d759",
    ROLE_FILES["ood_code"]: "a82642da15dea4c82d486b46f118a55e480e7613e011ed588caa647eed16b660",
    FACADE_SOURCE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    INDUCTION_SOURCE: "b2d43be8e260bbe4bfece494999d237d93258f676b19e2993eca09655e253e3a",
    MODEL_SOURCE: "49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2",
}

PREDICTION_TEXT = {
    "pred_a_valid_physical_instrument": "exact replay, live edits, supports, calls, sources, and rows",
    "pred_b_product_level_positive_control": "complete donor product transfers in every context",
    "pred_c_both_source_factors_fill_target_first": "both source factors fill target first slot",
    "pred_d_both_source_factors_fill_target_second": "both source factors fill target second slot",
    "pred_e_branch_exchangeable_downstream_family": "all four mappings pass natural and code contexts",
    "pred_f_donor_background_stability": "each mapping is stable with donor present and absent",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    parent_result = json.loads(PARENT_RESULT.read_text())
    if (
        parent_result.get("rung") != 532
        or parent_result.get("pred_a_exact_live_interaction_instrument") is not True
        or parent_result.get("pred_b_product_control_transfers") is not True
        or parent_result.get("pred_c_source_second_replaces_target_first") is not False
        or parent_result.get("pred_d_source_first_replaces_target_second") is not False
        or parent_result.get("pred_e_heldout_interaction_defined_factor") is not False
        or parent_result.get("pred_f_factor_replacements_compose") is not True
        or parent_result.get("strong_null") is not True
        or parent_result.get("frozen_scales") != {
            "alpha": parent.ALPHA, "beta": parent.BETA, "gamma": parent.GAMMA,
            "direct_alpha": parent.DIRECT_ALPHA, "direct_beta": parent.DIRECT_BETA,
        }
    ):
        raise RuntimeError("rung532 result authority changed")
    math_result = json.loads(MATH_RESULT.read_text())
    if (
        math_result.get("status") != "cpu_contract_passed"
        or math_result.get("all_four_parent_mappings_meet_base_bars") is not True
        or math_result.get("missing_parent_matched_controls") != [
            "source_first_to_target_first", "source_second_to_target_second"]
        or math_result.get("scientific_outcomes_opened") is not False
    ):
        raise RuntimeError("rung533 CPU contract changed")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    if (
        receipt.get("schema") != "induction_equality_tensor_final_ood_v2_rows_receipt"
        or receipt.get("status") != "frozen_before_any_v2_model_forward"
        or receipt.get("outcome_access") is not False
        or receipt.get("roles", {}).get("final_natural") != "one_shot_final"
        or receipt.get("roles", {}).get("ood_code") != "one_shot_code_ood"
    ):
        raise RuntimeError("row receipt authority changed")
    payloads, metadata = {}, {}
    for role in ROLES:
        entry = receipt.get("entries", {}).get(role, {})
        if entry.get("file_sha256") != HASHES[ROLE_FILES[role]]:
            raise RuntimeError(f"row entry changed for {role}")
        payload = torch.load(ROLE_FILES[role], map_location="cpu", weights_only=False)
        rows = payload.get("rows")
        cells = payload.get("copy_cells", {})
        if (
            payload.get("schema") != "induction_equality_tensor_final_ood_v2_role"
            or payload.get("role") != role
            or not isinstance(rows, torch.Tensor)
            or tuple(rows.shape) != (DOCUMENTS_PER_ROLE, TOKENS + 1)
            or rows.dtype != torch.long
            or any(
                not isinstance(cells.get(cell), torch.Tensor)
                or tuple(cells[cell].shape) != (DOCUMENTS_PER_ROLE, TOKENS)
                or cells[cell].dtype != torch.bool
                for cell in TASK_CELLS
            )
        ):
            raise RuntimeError(f"row payload shape changed for {role}")
        if bool(cells["positive"].logical_and(cells["matched_negative"]).any()):
            raise RuntimeError(f"positive and matched-negative cells overlap for {role}")
        half_support = {
            cell: [int(cells[cell][:DOCUMENT_SPLIT].sum()),
                   int(cells[cell][DOCUMENT_SPLIT:].sum())]
            for cell in TASK_CELLS
        }
        positive_documents = [
            int((cells["positive"][:DOCUMENT_SPLIT].sum(-1) > 0).sum()),
            int((cells["positive"][DOCUMENT_SPLIT:].sum(-1) > 0).sum()),
        ]
        if min(value for support in half_support.values() for value in support) <= 0 \
                or min(positive_documents) < 50:
            raise RuntimeError(f"insufficient frozen support for {role}")
        payloads[role] = payload
        metadata[role] = {
            "row_file_sha256": HASHES[ROLE_FILES[role]],
            "rows_tensor_sha256": entry.get("rows_tensor_sha256"),
            "half_support": half_support,
            "positive_documents": positive_documents,
        }
    return payloads, metadata


def replacement_pattern(arm: str, source_first, source_second, target_first, target_second):
    if arm == "native":
        return target_first * target_second
    if arm == "absent":
        return torch.zeros_like(target_first)
    if arm == "product_control":
        return parent.GAMMA * (source_first * source_second)
    permuted = arm.endswith("_key_control")
    mapping = arm.removesuffix("_key_control") if permuted else arm
    if mapping not in MAPPINGS:
        raise ValueError(f"unknown arm: {arm}")
    return math_contract.substitution(
        mapping, source_first, source_second, target_first, target_second,
        permuted=permuted,
    )


@torch.no_grad()
def run_forward(model, tokens, *, background: str, arm: str, direct: bool = False):
    if background not in BACKGROUNDS or arm not in ARMS:
        raise ValueError("unregistered background or arm")
    diagnostics = {
        "factor_reconstruction_max": 0.0, "branch_product_max_abs": 0.0,
        "donor_edit_rms": 0.0, "target_edit_rms": 0.0,
    }
    audit = {"native_attention": 0, "replayed_attention": 0, "native_mlp": 0}

    def attention(event):
        if direct or event.site not in parent.factor_parent.stage1.SITE_HEADS:
            write, next_value = event.block.attn(event.state, event.first_value)
            audit["native_attention"] += 1
            return write, next_value
        write, factors, support, reconstruction = parent.factor_parent._factor_site(
            event.state, event.first_value, event.block.attn, event.site, event.tokens)
        audit["replayed_attention"] += 1
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], reconstruction)
        if event.site == 8:
            first, second = parent.factor_screen._score_branches(event.state, event.block.attn)
            source = factors[parent.SOURCE_INDEX]
            target = factors[parent.TARGET_INDEX]
            source_first_native = first[:, parent.SOURCE_HEAD]
            source_second_native = second[:, parent.SOURCE_HEAD]
            target_first_native = first[:, parent.TARGET_HEAD]
            target_second_native = second[:, parent.TARGET_HEAD]
            causal = torch.tril(torch.ones(
                target_first_native.shape[-2:], dtype=torch.bool,
                device=target_first_native.device))
            native_product = parent.native_branch_product(
                target_first_native, target_second_native).masked_fill(~causal, 0.0)
            diagnostics["branch_product_max_abs"] = max(
                diagnostics["branch_product_max_abs"],
                float((native_product - target["p"]).abs().max()))
            source_first = source_first_native.float()
            source_second = source_second_native.float()
            target_first = target_first_native.float()
            target_second = target_second_native.float()
            if background == "donor_absent":
                write = write - source["native_term"]
                diagnostics["donor_edit_rms"] = float(
                    source["native_term"].float().square().mean().sqrt())
            if arm != "native":
                pattern = replacement_pattern(
                    arm, source_first, source_second, target_first, target_second)
                replacement = torch.bmm(pattern * support, target["u"])
                edit = replacement.to(write.dtype) - target["native_term"]
                write = write + edit
                diagnostics["target_edit_rms"] = float(edit.float().square().mean().sqrt())
        return write, event.first_value

    def mlp(event):
        audit["native_mlp"] += 1
        return event.block.mlp(event.state)

    logits = facade.forward_with_dispatch(model, tokens, attention, mlp, require_production=True)
    expected = ({"native_attention": 18, "replayed_attention": 0, "native_mlp": 18}
                if direct else
                {"native_attention": 15, "replayed_attention": 3, "native_mlp": 18})
    if audit != expected:
        raise RuntimeError(f"forward audit changed: {audit} != {expected}")
    return logits, diagnostics, audit


def empty_collection(payloads):
    return {
        "positive_document_sums": {
            role: torch.zeros(
                len(BACKGROUNDS), len(ARMS), DOCUMENTS_PER_ROLE, dtype=torch.float64)
            for role in ROLES
        },
        "positive_document_counts": {
            role: payloads[role]["copy_cells"]["positive"].sum(-1).double()
            for role in ROLES
        },
        "cell_sums": {
            role: torch.zeros(
                len(BACKGROUNDS), len(ARMS), 2, len(TASK_CELLS), dtype=torch.float64)
            for role in ROLES
        },
        "cell_counts": {
            role: torch.zeros(2, len(TASK_CELLS), dtype=torch.float64)
            for role in ROLES
        },
    }


def _accumulate_counts(collection, role, cells, start, stop):
    for half in range(2):
        half_start, half_stop = half * DOCUMENT_SPLIT, (half + 1) * DOCUMENT_SPLIT
        lo, hi = max(start, half_start), min(stop, half_stop)
        if lo >= hi:
            continue
        for cell_index, cell in enumerate(TASK_CELLS):
            collection["cell_counts"][role][half, cell_index] += int(cells[cell][lo:hi].sum())


def _accumulate_nll(collection, role, nll, cells, background_index, arm_index, start, stop):
    positive = cells["positive"][start:stop].to(nll.device)
    collection["positive_document_sums"][role][
        background_index, arm_index, start:stop] += (
            nll.mul(positive).sum(-1).double().cpu())
    for half in range(2):
        half_start, half_stop = half * DOCUMENT_SPLIT, (half + 1) * DOCUMENT_SPLIT
        lo, hi = max(start, half_start), min(stop, half_stop)
        if lo >= hi:
            continue
        local_lo, local_hi = lo - start, hi - start
        for cell_index, cell in enumerate(TASK_CELLS):
            selected = cells[cell][lo:hi].to(nll.device)
            collection["cell_sums"][role][background_index, arm_index, half, cell_index] += (
                nll[local_lo:local_hi][selected].double().sum().cpu())


@torch.no_grad()
def collect(model, payloads, *, smoke=False):
    collection = empty_collection(payloads)
    diagnostics = {
        "direct_native_calls": 0, "analytical_calls": 0,
        "native_replay_logit_max_abs": 0.0,
        "factor_reconstruction_max": 0.0, "branch_product_max_abs": 0.0,
        "minimum_donor_edit_rms": math.inf, "minimum_target_edit_rms": math.inf,
        "zero_intended_edits": 0, "roles_exercised": [],
    }
    roles = ROLES[:1] if smoke else ROLES
    device = next(model.parameters()).device
    for role in roles:
        rows = payloads[role]["rows"]
        cells = payloads[role]["copy_cells"]
        stop_document = BATCH if smoke else DOCUMENTS_PER_ROLE
        diagnostics["roles_exercised"].append(role)
        for start in range(0, stop_document, BATCH):
            stop = start + BATCH
            batch_rows = rows[start:stop]
            tokens = batch_rows[:, :-1].to(device)
            targets = batch_rows[:, 1:].to(device)
            direct_logits, _direct_diagnostics, _direct_audit = run_forward(
                model, tokens, background="donor_present", arm="native", direct=True)
            diagnostics["direct_native_calls"] += 1
            _accumulate_counts(collection, role, cells, start, stop)
            diagnostics["support_accumulator_exercised"] = True
            present_native_logits = None
            for background_index, background in enumerate(BACKGROUNDS):
                for arm_index, arm in enumerate(ARMS):
                    logits, row_diagnostics, _audit = run_forward(
                        model, tokens, background=background, arm=arm)
                    diagnostics["analytical_calls"] += 1
                    diagnostics["factor_reconstruction_max"] = max(
                        diagnostics["factor_reconstruction_max"],
                        row_diagnostics["factor_reconstruction_max"])
                    diagnostics["branch_product_max_abs"] = max(
                        diagnostics["branch_product_max_abs"],
                        row_diagnostics["branch_product_max_abs"])
                    if background == "donor_absent":
                        if row_diagnostics["donor_edit_rms"] <= 0:
                            diagnostics["zero_intended_edits"] += 1
                        else:
                            diagnostics["minimum_donor_edit_rms"] = min(
                                diagnostics["minimum_donor_edit_rms"],
                                row_diagnostics["donor_edit_rms"])
                    if arm != "native":
                        if row_diagnostics["target_edit_rms"] <= 0:
                            diagnostics["zero_intended_edits"] += 1
                        else:
                            diagnostics["minimum_target_edit_rms"] = min(
                                diagnostics["minimum_target_edit_rms"],
                                row_diagnostics["target_edit_rms"])
                    if background == "donor_present" and arm == "native":
                        present_native_logits = logits
                        diagnostics["native_replay_logit_max_abs"] = max(
                            diagnostics["native_replay_logit_max_abs"],
                            float((direct_logits - logits).abs().max()))
                    if not smoke:
                        nll = F.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                            reduction="none").view(BATCH, -1)
                        _accumulate_nll(
                            collection, role, nll, cells, background_index, arm_index,
                            start, stop)
                    if logits is not present_native_logits:
                        del logits
            if present_native_logits is None:
                raise RuntimeError("present/native replay was not exercised")
            del direct_logits, present_native_logits, tokens, targets
    expected_batches = 1 if smoke else len(ROLES) * BATCHES_PER_ROLE
    diagnostics["calls_exact"] = bool(
        diagnostics["direct_native_calls"] == expected_batches
        and diagnostics["analytical_calls"]
        == expected_batches * len(BACKGROUNDS) * len(ARMS))
    if not smoke:
        diagnostics["all_cell_supports_live"] = all(
            bool((collection["cell_counts"][role] > 0).all()) for role in ROLES)
        diagnostics["all_positive_document_supports_live"] = all(
            all(int((counts[half * DOCUMENT_SPLIT:(half + 1) * DOCUMENT_SPLIT] > 0).sum()) >= 50
                for half in range(2))
            for counts in collection["positive_document_counts"].values())
    return collection, diagnostics


def _vector_metrics(reference: torch.Tensor, candidate: torch.Tensor):
    return parent._vector_metrics(reference, candidate)


def analyze(collection):
    reports, contexts, effects = {}, [], {}
    for role in ROLES:
        counts = collection["positive_document_counts"][role]
        document_ce = (
            collection["positive_document_sums"][role]
            / counts[None, None].clamp_min(1))
        cell_ce = (
            collection["cell_sums"][role]
            / collection["cell_counts"][role][None, None].clamp_min(1))
        effects[role] = {}
        for background_index, background in enumerate(BACKGROUNDS):
            effects[role][background] = {}
            for half in range(2):
                start, stop = half * DOCUMENT_SPLIT, (half + 1) * DOCUMENT_SPLIT
                live = counts[start:stop] > 0
                native_docs = document_ce[
                    background_index, ARMS.index("native"), start:stop][live]
                absent_docs = document_ce[
                    background_index, ARMS.index("absent"), start:stop][live]
                native_effect = absent_docs - native_docs
                absent_positive = cell_ce[
                    background_index, ARMS.index("absent"), half, TASK_CELLS.index("positive")]
                native_positive = cell_ce[
                    background_index, ARMS.index("native"), half, TASK_CELLS.index("positive")]
                task_reference = float(absent_positive - native_positive)
                arm_reports, arm_effects = {}, {}
                for arm_index, arm in enumerate(ARMS):
                    arm_docs = document_ce[background_index, arm_index, start:stop][live]
                    effect = absent_docs - arm_docs
                    arm_effects[arm] = effect
                    positive = cell_ce[
                        background_index, arm_index, half, TASK_CELLS.index("positive")]
                    arm_reports[arm] = {
                        "positive_document_effect": _vector_metrics(native_effect, effect),
                        "positive_task_recovery": (
                            float(absent_positive - positive) / task_reference
                            if abs(task_reference) > 1e-30 else None),
                        "matched_negative_abs_mean_ce_change_from_native": abs(float(
                            cell_ce[background_index, arm_index, half,
                                    TASK_CELLS.index("matched_negative")]
                            - cell_ce[background_index, ARMS.index("native"), half,
                                      TASK_CELLS.index("matched_negative")])),
                        "off_target_abs_mean_ce_change_from_native": abs(float(
                            cell_ce[background_index, arm_index, half,
                                    TASK_CELLS.index("off_target")]
                            - cell_ce[background_index, ARMS.index("native"), half,
                                      TASK_CELLS.index("off_target")])),
                        "positive_documents": int(live.sum()),
                    }
                key = f"{role}/{background}/half{half}"
                reports[key] = {"arms": arm_reports}
                contexts.append(reports[key])
                effects[role][background][half] = arm_effects
    stability = {}
    for role in ROLES:
        for half in range(2):
            for mapping in MAPPINGS:
                key = f"{role}/half{half}/{mapping}"
                stability[key] = _vector_metrics(
                    effects[role]["donor_present"][half][mapping],
                    effects[role]["donor_absent"][half][mapping],
                )
    return reports, contexts, stability


def _base_holds(row, arm):
    report = row["arms"][arm]
    metric = report["positive_document_effect"]
    recovery = report["positive_task_recovery"]
    return bool(
        metric["cosine"] >= 0.85
        and metric["relative_error"] <= 0.60
        and recovery is not None and 0.65 <= recovery <= 1.40
        and report["matched_negative_abs_mean_ce_change_from_native"] <= 0.01
        and report["off_target_abs_mean_ce_change_from_native"] <= 0.01)


def _mapping_holds(row, mapping):
    metric = row["arms"][mapping]["positive_document_effect"]
    control = row["arms"][CONTROL_BY_MAPPING[mapping]]["positive_document_effect"]
    return bool(_base_holds(row, mapping) and metric["cosine"] >= control["cosine"] + 0.15)


def score(contexts, stability, diagnostics, checkpoint_hash):
    pred_a = bool(
        diagnostics["native_replay_logit_max_abs"] == 0.0
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["branch_product_max_abs"] == 0.0
        and diagnostics["minimum_donor_edit_rms"] > 0
        and diagnostics["minimum_target_edit_rms"] > 0
        and diagnostics["zero_intended_edits"] == 0
        and diagnostics["calls_exact"]
        and diagnostics["all_cell_supports_live"]
        and diagnostics["all_positive_document_supports_live"]
        and checkpoint_hash == facade.WEIGHTS_SHA256)
    mapping_passes = {
        mapping: [bool(pred_a and _mapping_holds(row, mapping)) for row in contexts]
        for mapping in MAPPINGS
    }
    pred_b = bool(pred_a and all(_base_holds(row, "product_control") for row in contexts))
    pred_c = bool(all(all(mapping_passes[mapping]) for mapping in (
        "source_first_to_target_first", "source_second_to_target_first")))
    pred_d = bool(all(all(mapping_passes[mapping]) for mapping in (
        "source_first_to_target_second", "source_second_to_target_second")))
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    pred_f = bool(pred_a and all(report["cosine"] >= 0.90 for report in stability.values()))
    predictions = dict(zip(PREDICTION_TEXT, (pred_a, pred_b, pred_c, pred_d, pred_e, pred_f)))
    checks = {
        "total_contexts": len(contexts),
        "product_control_contexts_passing": sum(
            _base_holds(row, "product_control") for row in contexts),
        "mapping_contexts_passing": {
            mapping: sum(values) for mapping, values in mapping_passes.items()
        },
        "background_stability_contexts_passing": sum(
            report["cosine"] >= 0.90 for report in stability.values()),
        "total_background_stability_contexts": len(stability),
    }
    return predictions, checks


def main():
    started = time.time()
    smoke = os.environ.get("RUNG533_SMOKE") == "1"
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert len(ARMS) == 11 and FORWARDS_PER_BATCH == 23 and FORWARDS == 2208
        assert len(MAPPINGS) == 4 and set(CONTROL_BY_MAPPING.values()).issubset(ARMS)
        print(json.dumps({
            "status": "dry_run_passed", "rung": RUNG, "model_loaded": False,
            "scientific_outcomes_opened": False, "forwards": FORWARDS,
            "roles": ROLES, "arms": ARMS, "backgrounds": BACKGROUNDS,
            "predictions": list(PREDICTION_TEXT),
        }, indent=2, sort_keys=True))
        return
    if not smoke and (OUT.exists() or BUNDLE.exists()):
        raise RuntimeError("rung533 output namespace already exists")
    payloads, input_metadata = validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    collection, diagnostics = collect(model, payloads, smoke=smoke)
    if smoke:
        instrument_pass = bool(
            diagnostics["calls_exact"]
            and diagnostics.get("support_accumulator_exercised") is True
            and diagnostics["roles_exercised"] == ["final_natural"]
            and diagnostics["native_replay_logit_max_abs"] == 0.0
            and diagnostics["factor_reconstruction_max"] <= 1e-10
            and diagnostics["branch_product_max_abs"] == 0.0
            and diagnostics["minimum_donor_edit_rms"] > 0
            and diagnostics["minimum_target_edit_rms"] > 0
            and diagnostics["zero_intended_edits"] == 0
            and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
        print(json.dumps({
            "status": "smoke_passed" if instrument_pass else "smoke_instrument_invalid",
            "instrument_pass": instrument_pass, "rung": RUNG,
            "scientific_outcomes_opened": False,
            "calls": diagnostics["direct_native_calls"] + diagnostics["analytical_calls"],
            "roles_exercised": diagnostics["roles_exercised"],
            "native_replay_logit_max_abs": diagnostics["native_replay_logit_max_abs"],
            "factor_reconstruction_max": diagnostics["factor_reconstruction_max"],
            "branch_product_max_abs": diagnostics["branch_product_max_abs"],
            "minimum_donor_edit_rms": diagnostics["minimum_donor_edit_rms"],
            "minimum_target_edit_rms": diagnostics["minimum_target_edit_rms"],
            "zero_intended_edits": diagnostics["zero_intended_edits"],
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }, indent=2, sort_keys=True))
        if not instrument_pass:
            raise RuntimeError("rung533 smoke instrument did not pass")
        return
    reports, contexts, stability = analyze(collection)
    predictions, checks = score(contexts, stability, diagnostics, checkpoint.weights_sha256)
    fixed_pairing = bool(
        predictions["pred_a_valid_physical_instrument"]
        and predictions["pred_b_product_level_positive_control"]
        and all(checks["mapping_contexts_passing"][mapping] == len(contexts) for mapping in (
            "source_second_to_target_first", "source_first_to_target_second"))
        and all(checks["mapping_contexts_passing"][mapping] < len(contexts) for mapping in (
            "source_first_to_target_first", "source_second_to_target_second")))
    product_only = bool(
        predictions["pred_a_valid_physical_instrument"]
        and predictions["pred_b_product_level_positive_control"]
        and not predictions["pred_c_both_source_factors_fill_target_first"]
        and not predictions["pred_d_both_source_factors_fill_target_second"])
    result = {
        "status": "completed", "rung": RUNG, **predictions,
        "fixed_pairing_outcome": fixed_pairing,
        "product_only_outcome": product_only,
        "reports": reports, "background_stability": stability,
        "checks": checks, "diagnostics": diagnostics,
        "pair": PAIR_NAME, "mappings": list(MAPPINGS),
        "frozen_scales": dict(math_contract.SCALES),
        "roles": list(ROLES), "document_halves": [[0, 96], [96, 192]],
        "task_cells": list(TASK_CELLS),
        "price": {
            "model_forwards": diagnostics["direct_native_calls"] + diagnostics["analytical_calls"],
            "backward_passes": 0, "fitted_vector_parameters": 0,
            "natural_documents": 192, "code_ood_documents": 192,
            "arms_per_background": len(ARMS),
        },
        "checkpoint": checkpoint, "input_metadata": input_metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "elapsed_seconds": time.time() - started,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "raw_tokens_logits_hidden_states_or_per_token_losses_included": False,
    }
    bundle = {
        "schema": "rung533_cross_corpus_document_ce_sufficient_statistics_v1",
        "collection": collection, "diagnostics": diagnostics,
        "raw_tokens_logits_hidden_states_or_per_token_losses_included": False,
        "rung532_census_rows_reused": False,
    }
    torch.save(bundle, BUNDLE)
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": RUNG, **predictions,
        "fixed_pairing_outcome": fixed_pairing, "product_only_outcome": product_only,
        "checks": checks, "calls": result["price"]["model_forwards"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
