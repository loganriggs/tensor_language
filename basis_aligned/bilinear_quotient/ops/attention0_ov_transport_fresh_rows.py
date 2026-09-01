"""RUNG 421 (Claude red-team lane) -- FRESH-ROW REPLICATION OF 419's pred_d TRANSPORT.

419/419b's only surviving prediction (native QK routed transport of the
task-interface U16 codebook) passed on razor margins: global routed R2
.70045 vs a >=.70 bar (+.0005), global-private gap .0555 vs >=.05, mean
consumer gap .0552 vs >=.05 -- all measured on the same SELECT rows used
throughout the 41x arc.  This rung rebuilds the identical deterministic
instrument from FIT and evaluates transport on 96 rows with ZERO row
overlap with 419's FIT, SELECT, or FINAL (mlp2_rank512_refit_v1
EVALUATION[0:96], file-sha pinned; overlap recomputed in-run, bar = 0).

Arms: {global codebook replacement, private codebook replacement, native}
via the unmodified base._transport on FRESH rows.

Frozen predictions
------------------
pred_a (instrument): fresh receipt file sha matches; in-run row-hash
    overlap with FIT+SELECT+FINAL == 0; payload fold <= 1e-10; refit
    codebook reproduces 419b's stored fit_global_sse 6105080715.752001
    and heldout global_advantage .019714322354085195 to rel 1e-6; fresh
    u16 edge relative squared error after measured remainder <= 1e-12.
pred_b (replication): fresh routed global R2 >= .65 AND within +-.05 of
    the 419 SELECT value .7004543544693138.
pred_c (margins survive): fresh (global - private) routed R2 gap >= .03
    AND fresh mean-consumer R2 gap >= .03 AND CE ordering preserved both
    waves (global wave damage <= private wave damage + .002; CE damage is
    CE ADDED ABOVE NATIVE -- LOWER IS BETTER).

Null: fresh global R2 < .60, or routed gap < .01, or CE ordering reversed
by > .002 on either wave => 419's pred_d margins were row-sampling
artifacts and the transport claim carries no adoption weight.

Price: screen only; no shipped object; attribution not compression; no
419 bar is altered by any outcome here.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import importlib.util
import json
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
QK = ROOT / "basis_aligned/qk_mdl"
OUT = BQ / "attention0_ov_transport_fresh_rows_results.json"
BASE = OPS / "attention0_ov_downstream_codebook.py"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
FRESH_RECEIPT = BQ / "mlp2_rank512_refit_v1_rows_receipt.json"
FRESH_ROLE = "EVALUATION"
FRESH_COUNT = 96
D = 1152
VOCAB = 50_257
RANK = 16
DOC_BATCH = 4
POSITIONS = tuple(range(16, 241, 16))

REF_FIT_GLOBAL_SSE = 6105080715.752001
REF_GLOBAL_ADVANTAGE = 0.019714322354085195
REF_SELECT_GLOBAL_R2 = 0.7004543544693138
REF_SELECT_PRIVATE_R2 = 0.6449369928112186


def _row_hashes(rows: torch.Tensor) -> set[str]:
    return {hashlib.sha256(row.contiguous().numpy().tobytes()).hexdigest()
            for row in rows.cpu()}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert RANK == 16 and FRESH_COUNT == 96 and len(POSITIONS) == 15
        assert FRESH_RECEIPT.exists() and BASE.exists() and ROWS_RECEIPT.exists()
        entries = json.loads(FRESH_RECEIPT.read_text())["entries"]
        assert FRESH_ROLE in entries and Path(entries[FRESH_ROLE]["path"]).exists()
        print("ATTENTION0 OV TRANSPORT FRESH ROWS | dry run: disjoint 96-row replication of 419 pred_d")
        return

    started = time.time()
    sys.path.insert(0, str(POLY))
    sys.path.insert(0, str(OPS))
    sys.path.insert(0, str(QK))
    spec = importlib.util.spec_from_file_location("ov_base", BASE)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)
    import bilin18_observed_model_facade as facade
    import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent
    import scoring
    from tier2_model import rope_tables, apply_rot

    device = torch.device("cuda")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    final_rows = rows_parent.load_role(receipt["entries"]["FINAL"])

    fresh_entry = json.loads(FRESH_RECEIPT.read_text())["entries"][FRESH_ROLE]
    fresh_path = Path(fresh_entry["path"])
    file_sha = hashlib.sha256(fresh_path.read_bytes()).hexdigest()
    fresh_all = torch.load(fresh_path, weights_only=True)
    fresh_rows = fresh_all[:FRESH_COUNT].contiguous()
    used = _row_hashes(fit_rows) | _row_hashes(select_rows) | _row_hashes(final_rows)
    overlap = sum(1 for h in _row_hashes(fresh_rows) if h in used)

    model, _ = facade.load_bilin18(device=device, dtype=torch.float32)
    block0 = model.transformer.h[0]
    vocab_ids = torch.arange(VOCAB, device=device)
    fit_mask = vocab_ids.remainder(5) != 4
    select_mask = ~fit_mask

    captured = base._capture_cproj_input(model, fit_rows, device).to(device)
    weight = block0.attn.c_proj.weight.detach().float()
    a_factor, b_factor = base._asvd(weight, captured)
    task_interface = torch.linalg.qr(
        a_factor[:, :RANK].float(), mode="reduced").Q.to(device)

    embedding = F.rms_norm(
        model.transformer.wte.weight.detach().float(), (D,))[:VOCAB]
    payload_fold_error = base._payload_exactness(model, embedding)

    write_samples = []
    for start in range(0, len(fit_rows), DOC_BATCH):
        tokens = fit_rows[start:start + DOC_BATCH, :-1].to(device)
        x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
        token_base = (block0.lambdas[0] + block0.lambdas[1]) * x0
        attention0, _ = block0.attn(F.rms_norm(token_base, (D,)), None)
        write_samples.append(attention0.float()[:, POSITIONS].reshape(-1, D))
    write_samples = torch.cat(write_samples)
    sigma = torch.sqrt(
        (write_samples @ task_interface).double().square().mean(0)).float()
    fit_gram, normalizers, _ = base._response_metric(
        model, fit_rows, task_interface, sigma, None, device)
    factor, _ = base._metric_factor(fit_gram)
    all_a = base._payload_codes(model, task_interface, embedding)
    codebook = base._fit_codebook(all_a, all_a, factor, fit_mask, select_mask)

    sse_rel = abs(codebook["fit_global_sse"] - REF_FIT_GLOBAL_SSE) / REF_FIT_GLOBAL_SSE
    adv_rel = abs(codebook["global_advantage"] - REF_GLOBAL_ADVANTAGE) / REF_GLOBAL_ADVANTAGE

    transport = base._transport(
        model, fresh_rows, device, task_interface, all_a, codebook, factor,
        rope_tables, apply_rot, scoring)

    fresh_global = transport["routed_u16_r2"]["global"]
    fresh_private = transport["routed_u16_r2"]["private"]
    routed_gap = fresh_global - fresh_private
    consumer_gap = (transport["mean_consumer_r2"]["global"]
                    - transport["mean_consumer_r2"]["private"])
    wave_deltas = [
        transport["ce"]["global"]["wave_damage"][wave]
        - transport["ce"]["private"]["wave_damage"][wave]
        for wave in range(2)]
    edge_rel = transport["u16_edge_relative_squared_error_after_measured_remainder"]

    pred_a = (
        file_sha == fresh_entry["file_sha256"]
        and overlap == 0
        and payload_fold_error <= 1e-10
        and sse_rel <= 1e-6
        and adv_rel <= 1e-6
        and edge_rel <= 1e-12)
    pred_b = fresh_global >= .65 and abs(fresh_global - REF_SELECT_GLOBAL_R2) <= .05
    pred_c = (routed_gap >= .03 and consumer_gap >= .03
              and all(delta <= .002 for delta in wave_deltas))
    null = (fresh_global < .60 or routed_gap < .01
            or any(delta > .002 for delta in wave_deltas))

    result = {
        "status": "attention0_ov_transport_fresh_rows_complete",
        "rung": 421,
        "claim_level": "fresh_row_replication_of_419_transport_screen_not_compression",
        "fresh_source": {"receipt": FRESH_RECEIPT.name, "role": FRESH_ROLE,
                         "count": FRESH_COUNT, "file_sha256": file_sha,
                         "row_overlap_with_419_roles": overlap},
        "instrument": {"payload_fold_max_abs": payload_fold_error,
                       "fit_global_sse": codebook["fit_global_sse"],
                       "fit_global_sse_rel_err": sse_rel,
                       "global_advantage": codebook["global_advantage"],
                       "global_advantage_rel_err": adv_rel,
                       "edge_rel_sq_after_remainder": edge_rel},
        "reference_419": {"select_global_r2": REF_SELECT_GLOBAL_R2,
                          "select_private_r2": REF_SELECT_PRIVATE_R2},
        "fresh_transport": {
            "routed_u16_r2": transport["routed_u16_r2"],
            "mean_consumer_r2": transport["mean_consumer_r2"],
            "consumer_r2": transport["consumer_r2"],
            "ce": transport["ce"],
            "routed_gap": routed_gap,
            "consumer_gap": consumer_gap,
            "wave_damage_deltas_global_minus_private": wave_deltas,
        },
        'pred_a_instrument_reproduces_419': bool(pred_a),
        'pred_b_fresh_row_transport_replicates': bool(pred_b),
        'pred_c_global_over_private_margins_survive': bool(pred_c),
        'null_419_pred_d_was_row_sampling_artifact': bool(null),
        "FINAL_opened": 0,
        "compression_or_adoption_licensed": False,
        "next_step": ("419_pred_d_carries_weight" if pred_a and pred_b and pred_c
                      and not null else "419_pred_d_no_adoption_weight"),
        "runtime_s": time.time() - started,
    }
    OUT.write_text(json.dumps(result, indent=1))
    print("wrote", OUT, f"({result['runtime_s']:.1f}s)")


if __name__ == "__main__":
    main()
