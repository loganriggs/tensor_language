#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_pattern_replay pred_b_native_pattern_tracks_shift pred_c_token_only_cue_state_loses_the_sense pred_d_context_suppresses_attention_on_agentive_by
"""Aspectual has/had definition-of-done battery, step 22: FOLD head 8.1's cue attention into token vs context.

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v21 (on natural rows the 8.1 shift scales
with its attention pattern on the cue; natural agentive `by` gets little attention).

WHY. The value 8.1 reads is token-only (block-0 value of the cue); v21 shows the SENSE is in the pattern
p = (q.k/D)(q2.k2/D). The key at the cue is k = rot(rms(W_k rms(live_8[cue]))) and live_8[cue] is an exact
lambda-weighted sum of writers: the embedding (token-only) plus attn/mlp 0-7 (context). Counterfactual fold:
recompute the cue key from the EMBEDDING writer alone (token-only cue state) and ask whether the resulting
pattern still tracks the removal shift across natural rows. If the token-only key loses the correlation and
gives agentive `by` rows MORE attention than native, the context writers at the cue position are what
suppress 8.1 on non-temporal `by` -- the sense enters through the cue's contextual state, not the query.

ROWS: v20 natural panel (32 since, 32 any-sense by; opened by v20/v21) and the v21 temporal-by panel.

PREDICTIONS (scored as written; failures preserved)
    pred_a_pattern_replay                 the manual native pattern equals the fold's pattern (v21 recipe) within
                                          1e-4 on every row
    pred_b_native_pattern_tracks_shift    Pearson r(p_native, shift) <= -0.40 on since rows and >= +0.40 on by rows
                                          (replay of v21's finding on the same rows)
    pred_c_token_only_cue_state_loses_the_sense   |Pearson r(p_token_key, shift)| <= 0.25 on the by rows
    pred_d_context_suppresses_attention_on_agentive_by   median p_token_key over the by rows >= 2 x median p_native
                                          over the by rows

PRICE (registered maximum): v20 panel, 2 batches: native 2 + 8.1 removal 2 + native-pattern fold 2 + two
counterfactual pattern passes x (trace 1 + capture 1) x 2 batches = 8 -> 14 forwards; 0 backwards; 0 fits. Bar <= 16.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_natural_v20 as v20
import run_aspectual_dod_bank_writer_fold_v13 as v13

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_cue_pattern_fold_v22_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_cue_pattern_fold_v22"
TOKENS = {"has": 468, "had": 550}
R_MIN, R_LOSS_MAX, SUPPRESS_RATIO, REPLAY_TOL = 0.40, 0.25, 2.0, 1e-4
FORWARDS_MAX = 16
HEAD81 = L.Component("attn8_h1_final", 8, "attn", (1,), "final")


def pearson(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    sx, sy = statistics.pstdev(x), statistics.pstdev(y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) * sx * sy)


def patterns(fw, rows, offsets, embed_only: bool):
    """Head 8.1 pattern on the cue at the final query, with the cue key from the native state or from the
    embedding-only cue state. Returns dict row_id -> pattern. One forward (trace) per batch, plus one
    captured attention input."""
    torch, F, model = fw.torch, fw.F, fw.model
    import jacclust.tt_model as TT
    attn = model.transformer.h[8].attn
    out = {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        tr = L.forward_trace_positions(fw, chunk, lambda r: (offsets[r.row_id],), upto_layer=8)
        captured = {}
        handle = attn.register_forward_pre_hook(lambda m, a: captured.__setitem__("x", a[0].detach().clone()))
        try:
            fw.forward(chunk)
        finally:
            handle.remove()
        x = captured["x"].clone()
        B, T, C = x.shape
        H, D = attn.n_head, attn.head_dim
        if embed_only:
            for i, row in enumerate(chunk):
                pos = offsets[row.row_id]
                Cw = v13.writer_contributions(tr[i], pos, upto=8)
                l0, l1 = float(model.transformer.h[8].lambdas[0]), float(model.transformer.h[8].lambdas[1])
                live_embed = l0 * Cw["embed"] + l1 * tr[i][("embed", pos)]
                x[i, pos] = F.rms_norm(live_embed.to(x.dtype), (C,))
        with torch.no_grad():
            q = attn.c_q(x).view(B, T, H, D); k = attn.c_k(x).view(B, T, H, D)
            q2 = attn.c_q2(x).view(B, T, H, D); k2 = attn.c_k2(x).view(B, T, H, D)
            cos, sin = attn.rotary(q)
            q, k = F.rms_norm(q, (D,)), F.rms_norm(k, (D,))
            q, k = TT.apply_rotary_emb(q, cos, sin), TT.apply_rotary_emb(k, cos, sin)
            q2, k2 = F.rms_norm(q2, (D,)), F.rms_norm(k2, (D,))
            q2, k2 = TT.apply_rotary_emb(q2, cos, sin), TT.apply_rotary_emb(k2, cos, sin)
            for i, row in enumerate(chunk):
                t, s = row.final, offsets[row.row_id]
                out[row.row_id] = float((q[i, t, 1].float() @ k[i, s, 1].float()) / D * (q2[i, t, 1].float() @ k2[i, s, 1].float()) / D)
    return out


def main() -> None:
    rows, sha = v20.load_rows()
    doc = json.loads(v20.ROWS.read_text())
    offsets = {hashlib.sha256(json.dumps([r["doc_index"], r["position"]]).encode()).hexdigest()[:24]: r["cue_offset"] for r in doc["rows"]}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": sha, "forwards_max": FORWARDS_MAX, "model_backwards": 0,
                          "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    fw.directions = L.readout_directions(backend.model, (HEAD81,), TOKENS["has"], TOKENS["had"])
    unit = {1: fw.directions[(HEAD81.name, 1)]}
    forwards = 0
    native, n = v1._run_arm(fw, rows); forwards += n
    alone, n = v1._run_arm(fw, rows, components=(HEAD81,), mode="project"); forwards += n
    fold = {}
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        out, lamb = L.head_source_terms_at(fw, chunk, 8, lambda r: r.final, unit); forwards += 1
        for row, entry in zip(chunk, out):
            fold[row.row_id] = entry[1]["pattern"][offsets[row.row_id]]
    p_native = patterns(fw, rows, offsets, embed_only=False); forwards += 2 * 2
    p_token = patterns(fw, rows, offsets, embed_only=True); forwards += 2 * 2
    replay = max(abs(p_native[r.row_id] - fold[r.row_id]) for r in rows)
    report = {}
    for cue in ("since", "by"):
        idx = [i for i, r in enumerate(rows) if r.construction == f"natural_{cue}"]
        d = [alone[i]["target_has_had"] - native[i]["target_has_had"] for i in idx]
        pn = [p_native[rows[i].row_id] for i in idx]; pt = [p_token[rows[i].row_id] for i in idx]
        report[cue] = {"n": len(idx), "r_native": pearson(pn, d), "r_token_key": pearson(pt, d),
                       "median_native": statistics.median(pn), "median_token_key": statistics.median(pt),
                       "mean_native": statistics.mean(pn), "mean_token_key": statistics.mean(pt)}
        print(cue, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in report[cue].items()})
    predictions = {
        "pred_a_pattern_replay": replay <= REPLAY_TOL,
        "pred_b_native_pattern_tracks_shift": report["since"]["r_native"] <= -R_MIN and report["by"]["r_native"] >= R_MIN,
        "pred_c_token_only_cue_state_loses_the_sense": abs(report["by"]["r_token_key"]) <= R_LOSS_MAX,
        "pred_d_context_suppresses_attention_on_agentive_by": report["by"]["median_token_key"] >= SUPPRESS_RATIO * report["by"]["median_native"],
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_cue_pattern_fold_result_v22", "candidate_id": CANDIDATE_ID, "rows_sha256": sha, "pattern_replay_max_abs": replay,
              "per_cue": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "replay": replay, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
