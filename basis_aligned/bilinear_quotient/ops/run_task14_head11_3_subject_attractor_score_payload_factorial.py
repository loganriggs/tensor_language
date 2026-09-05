#!/usr/bin/env python3
"""BASIC exact source score/value factorial for Task14 L11H3."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from pathlib import Path

import attention_source_factor_primitive as source_factor
import circuit_fast_screen_candidate_task14_select_cross_noun as authority


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits/fast_screens/task14_head11_3_subject_attractor_score_payload_factorial_v1_result.json"
PRIOR_ART_SHA256 = "9636504ee399f853a211a10529035d26f5ce89d3eb6847e28ed7c0d70ed95b45"
LAYER = 11
HEAD = 3
BATCH = 32
CONDITIONS = (
    "subject_score", "subject_payload", "subject_joint",
    "attractor_score", "attractor_payload", "attractor_joint", "complete_head",
)


def compile_plan():
    rows = authority.build_rows()
    return {
        "schema": "task14_head11_3_source_factor_plan_v1",
        "candidate_id": "subject_verb.number_agreement.head11_3_subject_attractor_score_payload_factorial",
        "split": "SELECT", "screen_tier": "BASIC", "row_count": len(rows),
        "authority_sha256": authority.validate_rows(rows),
        "site": {"layer": LAYER, "head": HEAD, "query": "final_prediction_position"},
        "sources": {"subject": 1, "attractor": "semantic_final_position"},
        "conditions": list(CONDITIONS),
        "price": {"model_forwards": 7, "example_evaluations": 576,
                  "backwards": 0, "parameter_updates": 0},
        "outcomes": ["donor_directed_is_are_margin", "donor_answer_ce"],
        "closed_splits": ["TEST", "OOD"],
        "limits": "Below-head BASIC localization only; no selectivity or OOD claim.",
    }


def _dependencies():
    import torch
    import torch.nn.functional as F
    os.environ.setdefault("BQLIB_NO_MODEL", "1")
    poly = ROOT.parent / "polynomial_causal"
    for path in (ROOT, ROOT / "ops", poly):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import bilin18_observed_model_facade as facade
    return torch, F, facade


def _native_logits(model, tokens, torch, F):
    x = model.transformer.wte(tokens)
    x = F.rms_norm(x, (x.size(-1),))
    x0, first = x, None
    for block in model.transformer.h:
        x, first = block(x, first, x0)
    logits = model.lm_head(F.rms_norm(x, (x.size(-1),)))
    return (30.0 * torch.tanh(logits / 30.0)).float()


def _pad(rows, side, length, torch, device):
    tokens = torch.full((len(rows), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, row in enumerate(rows):
        ids = row[f"{side}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, device=device)
        finals.append(len(ids) - 1)
    return tokens, torch.tensor(finals, device=device)


def _masked_head_delta(delta, mask, torch):
    """Select exact replacement rows; never permit fractional head scaling."""
    if mask is None:
        return delta
    if tuple(mask.shape) != (delta.shape[0],):
        raise RuntimeError("replacement head mask must have one entry per row")
    if mask.dtype != torch.bool:
        raise RuntimeError("replacement head mask must be boolean")
    if mask.device != delta.device:
        raise RuntimeError("replacement head mask must share the write device")
    return delta * mask.unsqueeze(-1)


def _same_batch_native_heads(replacement, native, mask, torch):
    """Use factors captured in this dispatch for designated reinstall rows."""
    if mask is None:
        return replacement
    if tuple(mask.shape) != (replacement.shape[0],) or replacement.shape != native.shape:
        raise RuntimeError("native reinstall mask or head shape is invalid")
    if mask.dtype != torch.bool or mask.device != replacement.device:
        raise RuntimeError("native reinstall mask must be boolean on the write device")
    return torch.where(mask.unsqueeze(-1), native.to(replacement.dtype), replacement)


def _factor_forward(model, tokens, finals, torch, F, facade, *, source_positions=None,
                    replacement_terms=None, replacement_heads=None,
                    replacement_head_mask=None, native_reinstall_mask=None):
    captured = {}

    def attention(event):
        if event.site != LAYER:
            return event.block.attn(event.state, event.first_value)
        write, factors = source_factor.replay_attention_with_source_factors(
            event.state, event.first_value, event.block.attn, finals, HEAD, torch, F,
        )
        captured.update({name: value.detach().clone() for name, value in factors.items()})
        if replacement_terms is not None:
            write = source_factor.install_source_terms(
                write, factors, finals, source_positions, replacement_terms, torch,
            )
        if replacement_heads is not None:
            rows = torch.arange(tokens.size(0), device=tokens.device)
            installed = _same_batch_native_heads(
                replacement_heads, factors["head"], native_reinstall_mask, torch)
            delta = (installed - factors["head"]).to(write.dtype)
            delta = _masked_head_delta(delta, replacement_head_mask, torch)
            write[rows, finals] += delta
        return write, event.first_value

    logits = facade.forward_with_dispatch(
        model, tokens, attention, lambda event: event.block.mlp(event.state), require_production=False,
    ).float()
    if set(captured) != {"p", "u", "head"}:
        raise RuntimeError("failed to capture exact attention source factors")
    return logits, captured


def _selected(factors, positions, torch):
    rows = torch.arange(len(positions), device=positions.device)
    return factors["p"][rows, positions], factors["u"][rows, positions]


def _condition_batch(tokens, finals, recipient, donor, recipient_attractor, donor_attractor, torch):
    batch = len(tokens)
    subject = torch.full((batch,), 1, dtype=torch.long, device=tokens.device)
    rp_s, ru_s = _selected(recipient, subject, torch)
    dp_s, du_s = _selected(donor, subject, torch)
    rp_a, ru_a = _selected(recipient, recipient_attractor, torch)
    dp_a, du_a = _selected(donor, donor_attractor, torch)
    native_subject = rp_s.unsqueeze(-1) * ru_s
    native_attractor = rp_a.unsqueeze(-1) * ru_a
    terms = (
        dp_s.unsqueeze(-1) * ru_s,
        rp_s.unsqueeze(-1) * du_s,
        dp_s.unsqueeze(-1) * du_s,
        dp_a.unsqueeze(-1) * ru_a,
        rp_a.unsqueeze(-1) * du_a,
        dp_a.unsqueeze(-1) * du_a,
        native_subject,
    )
    sources = (subject, subject, subject, recipient_attractor, recipient_attractor,
               recipient_attractor, subject)
    heads = tuple(recipient["head"] for _ in range(6)) + (donor["head"],)
    return (
        tokens.repeat(len(CONDITIONS), 1), finals.repeat(len(CONDITIONS)),
        torch.cat(sources), torch.cat(terms), torch.cat(heads),
    )


def _pair_metrics(logits, row, q, torch):
    donor, recipient = int(row["donor_answer_id"]), int(row["base_answer_id"])
    margin = float(logits[q, donor] - logits[q, recipient])
    ce = float(-torch.log_softmax(logits[q], dim=-1)[donor])
    return margin, ce


def score(evidence, replay_error, term_identity_error, native_positive_fraction):
    by_condition = {}
    complete = [item for item in evidence if item["condition"] == "complete_head"]
    complete_margin_delta = statistics.fmean(item["margin_delta"] for item in complete)
    complete_ce_gain = statistics.fmean(item["native_donor_ce"] - item["donor_ce"] for item in complete)
    for condition in CONDITIONS:
        cells = [item for item in evidence if item["condition"] == condition]
        margin_delta = statistics.fmean(item["margin_delta"] for item in cells)
        ce_gain = statistics.fmean(item["native_donor_ce"] - item["donor_ce"] for item in cells)
        by_condition[condition] = {
            "mean_margin_delta": margin_delta,
            "margin_direction_fraction": sum(item["margin_delta"] > 0 for item in cells) / len(cells),
            "margin_recovery_of_complete_head": margin_delta / complete_margin_delta,
            "mean_donor_ce_gain": ce_gain,
            "ce_recovery_of_complete_head": (
                ce_gain / complete_ce_gain if abs(complete_ce_gain) > 1e-12 else None
            ),
        }
    subject_score = by_condition["subject_score"]["margin_recovery_of_complete_head"]
    subject_payload = by_condition["subject_payload"]["margin_recovery_of_complete_head"]
    attractor = max(by_condition[name]["margin_recovery_of_complete_head"] for name in
                    ("attractor_score", "attractor_payload", "attractor_joint"))
    instrument = (native_positive_fraction >= .85 and replay_error <= 5e-5
                  and term_identity_error <= 5e-5 and complete_margin_delta > 0
                  and complete_ce_gain > 0
                  and by_condition["complete_head"]["margin_direction_fraction"] >= .75)
    pred_payload = instrument and subject_payload >= .25 and subject_payload >= subject_score + .10 \
        and subject_payload > attractor
    pred_score = instrument and subject_score >= .25 and subject_score >= subject_payload + .10 \
        and subject_score > attractor
    pred_attractor = instrument and attractor >= .25 and attractor >= max(subject_score, subject_payload)
    pred_neither = instrument and not (pred_payload or pred_score or pred_attractor)
    return {
        "native_positive_fraction": native_positive_fraction,
        "native_replay_max_absolute_logit_error": replay_error,
        "source_term_identity_max_absolute_error": term_identity_error,
        "conditions": by_condition,
        "predictions": {
            "pred_a_instrument_live": instrument,
            "pred_b_subject_payload": pred_payload,
            "pred_c_subject_score": pred_score,
            "pred_d_attractor_driven": pred_attractor,
            "pred_e_neither_or_other_source": pred_neither,
        },
    }


def evaluate(model, torch, F, facade):
    rows = authority.build_rows()
    device = next(model.parameters()).device
    length = max(max(len(row["base_ids"]), len(row["donor_ids"])) for row in rows)
    base_tokens, base_finals = _pad(rows, "base", length, torch, device)
    donor_tokens, donor_finals = _pad(rows, "donor", length, torch, device)
    native_all = _native_logits(model, torch.cat((base_tokens, donor_tokens)), torch, F)
    native_base = native_all[:len(rows)]
    evidence, replay_error, identity_error = [], 0.0, 0.0
    native_positive = []
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start+BATCH]
        bt, bf = base_tokens[start:start+BATCH], base_finals[start:start+BATCH]
        dt, df = donor_tokens[start:start+BATCH], donor_finals[start:start+BATCH]
        replay_b, base = _factor_forward(model, bt, bf, torch, F, facade)
        _replay_d, donor = _factor_forward(model, dt, df, torch, F, facade)
        replay_error = max(replay_error, float((replay_b - native_base[start:start+BATCH]).abs().max()))
        identity_error = max(identity_error,
                             float((torch.einsum("bk,bkd->bd", base["p"], base["u"])-base["head"]).abs().max()),
                             float((torch.einsum("bk,bkd->bd", donor["p"], donor["u"])-donor["head"]).abs().max()))
        recipient_attractor = bf
        donor_attractor = df
        expanded = _condition_batch(bt, bf, base, donor, recipient_attractor, donor_attractor, torch)
        patched, _ = _factor_forward(
            model, expanded[0], expanded[1], torch, F, facade,
            source_positions=expanded[2], replacement_terms=expanded[3], replacement_heads=expanded[4],
        )
        patched = patched.view(len(CONDITIONS), len(chunk), length, -1)
        for local, row in enumerate(chunk):
            q = int(bf[local])
            native_margin, native_donor_ce = _pair_metrics(replay_b[local], row, q, torch)
            native_correct = float(replay_b[local, q, int(row["base_answer_id"])]
                                   - replay_b[local, q, int(row["base_foil_id"])])
            native_positive.append(native_correct > 0)
            for condition_index, condition in enumerate(CONDITIONS):
                margin, donor_ce = _pair_metrics(patched[condition_index, local], row, q, torch)
                evidence.append({
                    "row_id": row["row_id"], "cell_id": row["cell_id"], "condition": condition,
                    "native_donor_margin": native_margin, "donor_margin": margin,
                    "margin_delta": margin - native_margin,
                    "native_donor_ce": native_donor_ce, "donor_ce": donor_ce,
                })
    return evidence, replay_error, identity_error, sum(native_positive) / len(native_positive)


def main():
    plan = compile_plan()
    if "--dry-run" in sys.argv or os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    torch, F, facade = _dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        evidence, replay, identity, capability = evaluate(model, torch, F, facade)
    scored = score(evidence, replay, identity, capability)
    predictions = scored["predictions"]
    if not predictions["pred_a_instrument_live"]:
        terminal = "invalid"
    elif predictions["pred_b_subject_payload"]:
        terminal = "subject_payload_screen"
    elif predictions["pred_c_subject_score"]:
        terminal = "subject_score_screen"
    elif predictions["pred_d_attractor_driven"]:
        terminal = "attractor_driven_screen"
    else:
        terminal = "neither_or_other_source_null"
    result = {
        "schema": "task14_head11_3_subject_attractor_score_payload_factorial_result_v1",
        "plan": plan, "prior_art_sha256": PRIOR_ART_SHA256,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "terminal": terminal, "score": scored, "evidence": evidence,
        "evaluated_splits": ["SELECT_BASIC"], "forbidden_splits_opened": [],
        "model_forwards": plan["price"]["model_forwards"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "score": scored}, indent=2))


if __name__ == "__main__":
    main()
