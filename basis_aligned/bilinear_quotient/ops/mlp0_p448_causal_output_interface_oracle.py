"""RUNG 409 -- P448 CAUSAL OUTPUT INTERFACE ORACLE.

Project exact native-minus-p448 output errors through the independent frozen
rank-64 MLP0 causal interface and through heldout joint/split output controls.
This is a future-native oracle ceiling, not an executable correction.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path("/workspace/tensor_language")
POLY = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
OPS = BQ / "ops"
OUT = BQ / "mlp0_p448_causal_output_interface_oracle_results.json"
BASIS_ARTIFACT = BQ / "joint_early_mlp_pca_composition_authoritative_v3_bases.pt"
LOSS_ARTIFACT = BQ / "mlp0_p448_router_oracle_losses.pt"
PARENT = BQ / "mlp0_p448_router_oracle_ceiling_results.json"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
CONFIRM_RECEIPT = BQ / "mlp0_c512_mlp1_interchange_v1_rows_receipt.json"
CONFIRM_CACHE = BQ / ".rowcache_mlp0_c512_mlp1_interchange_v1/eval_384_source_documents.pt"
FIT_CACHE = BQ / ".rowcache/fineweb_n192_skip11000.pt"
BASIS_FILE_SHA = "0eee01f39087548a479486d068404f78c4bdc2fd930932add162212da31fe4d9"
B0_BASIS_SHA = "cb57c81a5b5ecbe8a1ad0f13f0f8b9e9df20d01ea237399db40659da75ab4b52"
LOSS_TENSOR_SHA = "e6d92614ad4fbe5b6e63aa2939e7df6ecb197281c114aae9696baf3bc68ab082"
FIT_SLICE = (0, 24)
D = 1152
H = 4608
RANKS = (448, 640, 768)
SOURCE_DOCUMENTS = 384
TRAIN_DOCUMENTS = 192
EVAL_DOCUMENTS = 192
WAVE_DOCUMENTS = 96
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)
P448_VALUES = 9_954_432
P640_VALUES = 11_945_088
P768_VALUES = 13_272_192
B0_INTERFACE_VALUES = 153_920
P448_B0_VALUES = P448_VALUES + B0_INTERFACE_VALUES
ARMS = (
    "NATIVE", "P448", "P640", "P768",
    "B0_T", "B0_I", "B0_TI", "B0_FULL",
    "JOINT_TI_64", "SPLIT_T32_I32", "TOTAL_ERROR_64", "RANDOM_64",
)


def _file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def _project(value, basis):
    original = value.shape
    flat = value.float().reshape(-1, D)
    return ((flat @ basis) @ basis.T).reshape(original)


def _top_basis(gram, rank):
    gram = 0.5 * (gram + gram.T)
    values, vectors = torch.linalg.eigh(gram)
    basis = vectors[:, -rank:].contiguous()
    retained = float(values[-rank:].double().sum() / values.clamp_min(0).double().sum())
    orthogonality = float((basis.T @ basis - torch.eye(rank, device=basis.device)).abs().max())
    return basis, retained, orthogonality


def _relative_mse(numerator, denominator):
    return float(numerator / max(denominator, 1e-30))


@torch.no_grad()
def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert all(path.exists() for path in (
            BASIS_ARTIFACT, LOSS_ARTIFACT, PARENT, ROWS_RECEIPT,
            CONFIRM_RECEIPT, CONFIRM_CACHE, FIT_CACHE))
        assert TRAIN_DOCUMENTS + EVAL_DOCUMENTS == SOURCE_DOCUMENTS
        assert len(ARMS) == 12 and P448_B0_VALUES == 10_108_352
        assert P448_B0_VALUES < P640_VALUES < P768_VALUES
        print("MLP0 p448 OUTPUT INTERFACE | dry run: authorities, split, arms, prices, bars valid")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(BQ.parent / "qk_mdl"))
    import bilin18_observed_model_facade as facade
    import mlp0_centered_context_anova_factorial as base
    import mlp0_rank448_branch_error_factorial as rung403
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    from mlp0_context_metric_shared_input_frontier import _covariance
    from mlp_late_context_metric_shared_input_screen import _rrr_program
    from mlp_shared_input_svd_all_layers_screen import _manual_logits

    device = torch.device("cuda")
    parent = json.loads(PARENT.read_text())
    if not parent["null_router_oracle_has_no_adoption_headroom"]:
        raise RuntimeError("rung407 null does not license the output-interface ceiling")

    receipt = json.loads(CONFIRM_RECEIPT.read_text())
    current_rows_receipt = json.loads(ROWS_RECEIPT.read_text())
    all_rows = torch.load(CONFIRM_CACHE, map_location="cpu")
    records = receipt["document_provenance"]["sets"]["eval"]
    chunk0 = [index for index, record in enumerate(records) if int(record["chunk_id"]) == 0]
    ordinals = [int(records[index]["source_document_ordinal"]) for index in chunk0]
    document_ids = [records[index]["document_id"] for index in chunk0]
    rows = all_rows[chunk0, :257].long().contiguous()
    population_exact = (
        tuple(all_rows.shape) == (585, 513)
        and tuple(rows.shape) == (SOURCE_DOCUMENTS, 257)
        and ordinals == list(range(SOURCE_DOCUMENTS))
        and len(set(document_ids)) == SOURCE_DOCUMENTS
        and _tensor_sha256(all_rows) == receipt["entries"]["eval"]["tensor_raw_sha256"]
        and all(receipt["disjointness_gates"].values())
        and all(current_rows_receipt["disjointness"].values()))

    basis_bundle = torch.load(BASIS_ARTIFACT, map_location="cpu", weights_only=True)
    b0_cpu = basis_bundle["sites"][0]["basis"].float().contiguous()
    b0_hash = _tensor_sha256(b0_cpu)
    basis_authority_exact = (
        _file_sha256(BASIS_ARTIFACT) == BASIS_FILE_SHA
        and tuple(b0_cpu.shape) == (D, 64)
        and b0_hash == B0_BASIS_SHA
        and basis_bundle["sites"][0]["basis_sha256"] == B0_BASIS_SHA)

    loss_bundle = torch.load(LOSS_ARTIFACT, map_location="cpu", weights_only=True)
    saved_losses = loss_bundle["losses"].float()
    saved_names = tuple(loss_bundle["program_names"])
    loss_authority_exact = (
        tuple(saved_losses.shape) == (8, SOURCE_DOCUMENTS, 192)
        and _tensor_sha256(saved_losses) == LOSS_TENSOR_SHA)

    cached = torch.load(FIT_CACHE, map_location="cpu")
    cached = cached["rows"] if isinstance(cached, dict) else cached
    program_fit_rows = cached[FIT_SLICE[0]:FIT_SLICE[1], :257].long().contiguous()
    source_model, source_checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    covariance = _covariance(source_model, program_fit_rows, _manual_logits)
    programs = {}
    program_diagnostics = {}
    for rank in RANKS:
        program, _basis, diagnostic = _rrr_program(
            source_model.transformer.h[0].mlp, covariance, rank=rank)
        programs[rank] = {name: value.detach().clone() for name, value in program.items()}
        program_diagnostics[rank] = diagnostic
    del covariance, _basis, source_model
    torch.cuda.empty_cache()

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    native_factors = {
        "left": model.transformer.h[0].mlp.Left.weight.detach().float(),
        "right": model.transformer.h[0].mlp.Right.weight.detach().float(),
        "down": model.transformer.h[0].mlp.Down.weight.detach().float(),
    }
    compact_factors = {name: value.detach().float() for name, value in programs[448].items()}
    fit_rows = rows_parent.load_role(current_rows_receipt["entries"]["FIT"])
    token_cpu, context_cpu, gain_cpu = base._capture_inputs(model, fit_rows, device)
    native_reference = rung403._reference_for_factors(
        token_cpu, context_cpu, gain_cpu, native_factors, device)
    compact_reference = rung403._reference_for_factors(
        token_cpu, context_cpu, gain_cpu, compact_factors, device)

    identity = {"native_num": 0.0, "native_den": 0.0,
                "compact_num": 0.0, "compact_den": 0.0,
                "branch_closure_max_abs": 0.0}

    def errors_for(event, token_base, native):
        compact = rung403._compact_deployed(event.state, compact_factors, native.dtype)
        retained_n, branches_n, _ = rung403._exact_components(
            base, token_base, event.attention_write, event.state,
            native_reference, native_factors)
        retained_p, branches_p, _ = rung403._exact_components(
            base, token_base, event.attention_write, event.state,
            compact_reference, compact_factors)
        direct_n = rung403._bilinear(event.state, event.state, native_factors)
        direct_p = rung403._bilinear(event.state, event.state, compact_factors)
        analytical_n = retained_n + sum(branches_n.values())
        analytical_p = retained_p + sum(branches_p.values())
        identity["native_num"] += float((analytical_n.double() - direct_n.double()).square().sum())
        identity["native_den"] += float(direct_n.double().square().sum())
        identity["compact_num"] += float((analytical_p.double() - direct_p.double()).square().sum())
        identity["compact_den"] += float(direct_p.double().square().sum())
        branch_errors = {name: branches_n[name] - branches_p[name] for name in ("T", "C", "I", "S")}
        total = native.float() - compact.float()
        named_sum64 = sum(value.double() for value in branch_errors.values())
        branch_errors["A"] = total.double() - named_sum64
        closure = total.double() - named_sum64 - branch_errors["A"]
        identity["branch_closure_max_abs"] = max(
            identity["branch_closure_max_abs"], float(closure.abs().max()))
        return compact, branch_errors, total

    train_grams = {
        name: torch.zeros((D, D), device=device, dtype=torch.float32)
        for name in ("T", "I", "TI", "TOTAL")
    }
    train_calls = 0
    for start in range(0, TRAIN_DOCUMENTS, DOCUMENT_BATCH):
        batch = rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        block0 = model.transformer.h[0]
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token

        def attention(event):
            return event.block.attn(event.state, event.first_value)

        def mlp(event):
            nonlocal train_calls
            native = event.block.mlp(event.state)
            if event.site != 0:
                return native
            train_calls += 1
            _compact, branch_errors, total = errors_for(event, token_base, native)
            values = {
                "T": branch_errors["T"],
                "I": branch_errors["I"],
                "TI": branch_errors["T"] + branch_errors["I"],
                "TOTAL": total,
            }
            for name, value in values.items():
                flat = value[:, SCORING].reshape(-1, D).float()
                train_grams[name].add_(flat.T @ flat)
            return native

        facade.forward_with_dispatch(model, tokens, attention, mlp)

    joint_ti, joint_retained, joint_orth = _top_basis(train_grams["TI"], 64)
    split_t, split_t_retained, split_t_orth = _top_basis(train_grams["T"], 32)
    split_i, split_i_retained, split_i_orth = _top_basis(train_grams["I"], 32)
    total_error, total_retained, total_orth = _top_basis(train_grams["TOTAL"], 64)
    generator = torch.Generator(device=device).manual_seed(409)
    random_basis, _ = torch.linalg.qr(torch.randn(D, 64, generator=generator, device=device))
    random_orth = float((random_basis.T @ random_basis - torch.eye(64, device=device)).abs().max())
    b0 = b0_cpu.to(device)
    b0_orth = float((b0.T @ b0 - torch.eye(64, device=device)).abs().max())

    loss_parts = {name: [] for name in ARMS}
    calls = {name: {"forwards": 0, "attention": 0, "site0": 0, "other_mlp": 0} for name in ARMS}
    state_replay_max = 0.0
    mse = {
        "joint_T_num": 0.0, "joint_I_num": 0.0,
        "split_T_num": 0.0, "split_I_num": 0.0,
        "T_den": 0.0, "I_den": 0.0,
    }

    eval_rows = rows[TRAIN_DOCUMENTS:]
    for start in range(0, EVAL_DOCUMENTS, DOCUMENT_BATCH):
        batch = eval_rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        raw_token = F.rms_norm(model.transformer.wte(tokens), (D,))
        block0 = model.transformer.h[0]
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * raw_token
        cache = {}

        for arm in ARMS:
            def attention(event, arm=arm):
                calls[arm]["attention"] += 1
                return event.block.attn(event.state, event.first_value)

            def mlp(event, arm=arm):
                nonlocal state_replay_max
                native = event.block.mlp(event.state)
                if event.site != 0:
                    calls[arm]["other_mlp"] += 1
                    return native
                calls[arm]["site0"] += 1
                if not cache:
                    compact, branch_errors, total = errors_for(event, token_base, native)
                    cache.update({
                        "state": event.state.detach().clone(),
                        "native": native.detach().clone(),
                        "compact": compact.detach().clone(),
                        "errors": {name: value.detach().clone() for name, value in branch_errors.items()},
                        "total": total.detach().clone(),
                    })
                    t = branch_errors["T"][:, SCORING]
                    i = branch_errors["I"][:, SCORING]
                    jt, ji = t - _project(t, joint_ti), i - _project(i, joint_ti)
                    st, si = t - _project(t, split_t), i - _project(i, split_i)
                    mse["joint_T_num"] += float(jt.double().square().sum())
                    mse["joint_I_num"] += float(ji.double().square().sum())
                    mse["split_T_num"] += float(st.double().square().sum())
                    mse["split_I_num"] += float(si.double().square().sum())
                    mse["T_den"] += float(t.double().square().sum())
                    mse["I_den"] += float(i.double().square().sum())
                else:
                    state_replay_max = max(
                        state_replay_max, float((event.state - cache["state"]).abs().max()))

                if arm == "NATIVE":
                    result = native
                elif arm == "P448":
                    result = cache["compact"]
                elif arm == "P640":
                    result = rung403._compact_deployed(event.state, programs[640], native.dtype)
                elif arm == "P768":
                    result = rung403._compact_deployed(event.state, programs[768], native.dtype)
                elif arm == "B0_T":
                    result = cache["compact"].float() + _project(cache["errors"]["T"], b0)
                elif arm == "B0_I":
                    result = cache["compact"].float() + _project(cache["errors"]["I"], b0)
                elif arm == "B0_TI":
                    value = cache["errors"]["T"] + cache["errors"]["I"]
                    result = cache["compact"].float() + _project(value, b0)
                elif arm == "B0_FULL":
                    result = cache["compact"].float() + _project(cache["total"], b0)
                elif arm == "JOINT_TI_64":
                    value = cache["errors"]["T"] + cache["errors"]["I"]
                    result = cache["compact"].float() + _project(value, joint_ti)
                elif arm == "SPLIT_T32_I32":
                    result = (cache["compact"].float()
                              + _project(cache["errors"]["T"], split_t)
                              + _project(cache["errors"]["I"], split_i))
                elif arm == "TOTAL_ERROR_64":
                    result = cache["compact"].float() + _project(cache["total"], total_error)
                elif arm == "RANDOM_64":
                    result = cache["compact"].float() + _project(cache["total"], random_basis)
                else:
                    raise RuntimeError(arm)
                return result.to(native.dtype)

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            loss = F.cross_entropy(
                logits[:, SCORING].transpose(1, 2), targets[:, SCORING], reduction="none")
            loss_parts[arm].append(loss.cpu())
            calls[arm]["forwards"] += 1

    losses = torch.stack([torch.cat(loss_parts[name]) for name in ARMS])
    native_ce = float(losses[0].double().mean())
    damage = {name: float(losses[index].double().mean()) - native_ce
              for index, name in enumerate(ARMS)}
    wave_damage = {}
    for arm_index, arm in enumerate(ARMS):
        wave_damage[arm] = []
        for wave in range(2):
            left, right = wave * WAVE_DOCUMENTS, (wave + 1) * WAVE_DOCUMENTS
            wave_native = float(losses[0, left:right].double().mean())
            wave_damage[arm].append(float(losses[arm_index, left:right].double().mean()) - wave_native)

    saved_index = {name: index for index, name in enumerate(saved_names)}
    baseline_saved_error = {
        "NATIVE": float((losses[ARMS.index("NATIVE")] - saved_losses[
            saved_index["native"], TRAIN_DOCUMENTS:]).abs().max()),
        "P448": float((losses[ARMS.index("P448")] - saved_losses[
            saved_index["covariance_p448"], TRAIN_DOCUMENTS:]).abs().max()),
        "P640": float((losses[ARMS.index("P640")] - saved_losses[
            saved_index["covariance_p640"], TRAIN_DOCUMENTS:]).abs().max()),
        "P768": float((losses[ARMS.index("P768")] - saved_losses[
            saved_index["covariance_p768"], TRAIN_DOCUMENTS:]).abs().max()),
    }
    expected_calls = {
        "forwards": EVAL_DOCUMENTS // DOCUMENT_BATCH,
        "attention": 18 * EVAL_DOCUMENTS // DOCUMENT_BATCH,
        "site0": EVAL_DOCUMENTS // DOCUMENT_BATCH,
        "other_mlp": 17 * EVAL_DOCUMENTS // DOCUMENT_BATCH,
    }
    calls_live = all(value == expected_calls for value in calls.values())
    branch_identity = {
        "native_relative_mse": _relative_mse(identity["native_num"], identity["native_den"]),
        "compact_relative_mse": _relative_mse(identity["compact_num"], identity["compact_den"]),
        "closure_max_abs": identity["branch_closure_max_abs"],
    }
    output_mse = {
        "joint_T": _relative_mse(mse["joint_T_num"], mse["T_den"]),
        "joint_I": _relative_mse(mse["joint_I_num"], mse["I_den"]),
        "split_T": _relative_mse(mse["split_T_num"], mse["T_den"]),
        "split_I": _relative_mse(mse["split_I_num"], mse["I_den"]),
    }
    joint_geometric = math.sqrt(output_mse["joint_T"] * output_mse["joint_I"])
    split_geometric = math.sqrt(output_mse["split_T"] * output_mse["split_I"])
    split_mse_gain = 1 - split_geometric / max(joint_geometric, 1e-30)

    exact_programs = all(
        int(program_diagnostics[rank]["rank"]) == rank
        and {name: tuple(value.shape) for name, value in programs[rank].items()} == {
            "encoder": (rank, D), "left": (H, rank), "right": (H, rank),
            "down": (D, H), "bias": (D,)}
        for rank in RANKS)
    all_orthogonal = max(
        b0_orth, joint_orth, split_t_orth, split_i_orth, total_orth, random_orth) <= 1e-4
    pred_a = (
        population_exact and basis_authority_exact and loss_authority_exact
        and exact_programs and all_orthogonal and calls_live and train_calls == TRAIN_DOCUMENTS // DOCUMENT_BATCH
        and state_replay_max == 0.0 and max(baseline_saved_error.values()) == 0.0
        and branch_identity["native_relative_mse"] <= 1e-8
        and branch_identity["compact_relative_mse"] <= 1e-8
        and branch_identity["closure_max_abs"] == 0.0
        and bool(torch.isfinite(losses).all()))
    b0_gain = damage["P448"] - damage["B0_FULL"]
    pred_b = (
        b0_gain >= .30 * damage["P448"]
        and all(wave_damage["P448"][wave] - wave_damage["B0_FULL"][wave] >= .001
                for wave in range(2))
        and damage["RANDOM_64"] - damage["B0_FULL"] >= .001)
    ti_gain = damage["P448"] - damage["B0_TI"]
    pred_c = (
        ti_gain >= .70 * b0_gain
        and damage["P448"] - damage["B0_T"] >= .0002
        and damage["P448"] - damage["B0_I"] >= .0002
        and all(wave_damage[arm][wave] <= wave_damage["P448"][wave] + .0005
                for arm in ("B0_T", "B0_I") for wave in range(2)))
    pred_d = (
        split_mse_gain >= .10
        and damage["JOINT_TI_64"] - damage["SPLIT_T32_I32"] >= .0002
        and all(wave_damage["SPLIT_T32_I32"][wave] <= wave_damage["JOINT_TI_64"][wave]
                for wave in range(2)))
    learned = ("B0_FULL", "TOTAL_ERROR_64", "JOINT_TI_64", "SPLIT_T32_I32")
    strong_null = (
        not pred_a or b0_gain < .0002
        or damage["RANDOM_64"] - damage["B0_FULL"] < .0002
        or max(damage["P448"] - damage[name] for name in learned) < .0002)

    result = {
        "status": "mlp0_p448_causal_output_interface_oracle_complete",
        "rung": 409,
        "claim_level": "heldout_future_native_output_repair_ceiling_not_executable",
        "convention": "CE added above native; lower is better",
        "authority": {
            "source_documents": SOURCE_DOCUMENTS,
            "train_documents_half_open": [0, TRAIN_DOCUMENTS],
            "evaluation_documents_half_open": [TRAIN_DOCUMENTS, SOURCE_DOCUMENTS],
            "evaluation_waves": [[192, 288], [288, 384]],
            "scoring_positions_half_open": [SCORING.start, SCORING.stop],
            "population_exact": population_exact,
            "basis_file_sha256": _file_sha256(BASIS_ARTIFACT),
            "b0_basis_sha256": b0_hash,
            "basis_authority_exact": basis_authority_exact,
            "loss_tensor_sha256": _tensor_sha256(saved_losses),
            "loss_authority_exact": loss_authority_exact,
            "baseline_saved_loss_max_abs_error": baseline_saved_error,
            "FINAL_opened": 0,
        },
        "programs": {
            "source_checkpoint": source_checkpoint.__dict__,
            "evaluation_checkpoint": checkpoint.__dict__,
            "diagnostics": {str(rank): value for rank, value in program_diagnostics.items()},
            "exact": exact_programs,
        },
        "branch_identity": branch_identity,
        "basis_diagnostics": {
            "orthogonality_max_abs": {
                "B0_64": b0_orth, "JOINT_TI_64": joint_orth,
                "SPLIT_T32": split_t_orth, "SPLIT_I32": split_i_orth,
                "TOTAL_ERROR_64": total_orth, "RANDOM_64": random_orth,
            },
            "training_retained_energy": {
                "JOINT_TI_64": joint_retained, "SPLIT_T32": split_t_retained,
                "SPLIT_I32": split_i_retained, "TOTAL_ERROR_64": total_retained,
            },
            "heldout_branch_relative_mse": output_mse,
            "joint_geometric_mse": joint_geometric,
            "split_geometric_mse": split_geometric,
            "split_relative_gain": split_mse_gain,
        },
        "physical": {
            "native_ce": native_ce,
            "damage": damage,
            "wave_damage": wave_damage,
            "calls": calls,
            "calls_live": calls_live,
            "training_site0_calls": train_calls,
            "state_replay_max_abs_error": state_replay_max,
        },
        "literal_price": {
            "p448_values": P448_VALUES,
            "historical_b0_interface_values": B0_INTERFACE_VALUES,
            "optimistic_p448_plus_b0_values": P448_B0_VALUES,
            "p640_values": P640_VALUES,
            "p768_values": P768_VALUES,
            "oracle_reads_native_error_and_is_not_executable": True,
            "new_basis_producers_and_operations_not_priced": True,
        },
        "b0_full_gain_over_p448": b0_gain,
        "b0_ti_share_of_full_gain": ti_gain / b0_gain if b0_gain > 0 else 0.0,
        'pred_a_authority_programs_branches_and_physical_paths_are_exact': bool(pred_a),
        'pred_b_frozen_causal_interface_contains_current_p448_error': bool(pred_b),
        'pred_c_frozen_interface_gain_is_token_grammar_led': bool(pred_c),
        'pred_d_separate_token_and_interaction_output_bases_win': bool(pred_d),
        "null_rank64_output_repair_has_no_specific_heldout_gain": bool(strong_null),
        "next_object": (
            "physical_historical_B_l5_r64_on_p448" if pred_a and pred_b and pred_c and not strong_null
            else "separate_T_I_output_producers" if pred_a and pred_d and not strong_null
            else "shared_rank64_output_producer" if pred_a and not strong_null
            else None),
        "executable_repair_or_adoption_licensed": False,
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("MLP0 p448 CAUSAL OUTPUT INTERFACE ORACLE DONE", flush=True)


if __name__ == "__main__":
    main()
