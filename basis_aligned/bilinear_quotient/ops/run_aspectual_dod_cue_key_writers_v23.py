#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_key_writer_closure pred_b_few_writers_build_the_key pred_c_same_writers_on_templates_and_corpus
"""Aspectual has/had definition-of-done battery, step 23: which writers build head 8.1's cue KEY?

Lane: Claude circuit lane. Library: `aspectual_dod_lib.py`. Parent: v22 (the cue key is contextual: with the
embedding-only cue state 8.1 attends ~zero; the block-0 value is the token-only part).

WHY. Last fold before the kill criterion on this port. The score s1 = q . k with k = rot(rms(W_k rms(live_8
[cue]))) is linear in live_8[cue] once the two RMS scalars are taken from the native state, and live_8[cue]
is an exact lambda-weighted sum of writers; same for s2. So each score splits exactly into writer terms.
Report writer shares of the two scores, pooled over rows where the native pattern is in the top half (the
rows where 8.1 actually attends), on the 64 discovery template rows and the 64 natural rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_key_writer_closure              writer sums reproduce both native scores within relative 1e-3 on
                                           every row
    pred_b_few_writers_build_the_key       on the discovery rows, the top four writers carry >= 0.70 of BOTH
                                           scores (pooled shares). Prior: unsure (kill criterion: <= 4 writers)
    pred_c_same_writers_on_templates_and_corpus   the top-two writers of s1 and of s2 on the natural rows are
                                           within the top four on the discovery rows

PRICE (registered maximum): 4 batches x (trace 1 + capture 1) = 8 forwards; 0 backwards; 0 fits. Bar <= 10.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
import time
from pathlib import Path

import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_aspectual_dod_natural_v20 as v20
import run_aspectual_dod_bank_writer_fold_v13 as v13

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/aspectual_anchor_dod_cue_key_writers_v23_result.json"
CANDIDATE_ID = "aspectual_anchor.has_vs_had.dod_cue_key_writers_v23"
CLOSURE_TOL, TOP4_MIN = 1e-3, 0.70
FORWARDS_MAX = 10
WRITERS = ["embed"] + [f"{k}:{l:02d}" for l in range(8) for k in ("attn", "mlp")]


def key_terms(fw, rows, offsets):
    """Per row: writer terms of s1 and s2 at (final query, cue key) for head 8.1, plus native scores."""
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
        x = captured["x"]
        B, T, C = x.shape
        H, D = attn.n_head, attn.head_dim
        with torch.no_grad():
            q = attn.c_q(x).view(B, T, H, D); q2 = attn.c_q2(x).view(B, T, H, D)
            k = attn.c_k(x).view(B, T, H, D); k2 = attn.c_k2(x).view(B, T, H, D)
            cos, sin = attn.rotary(q)
            qn, q2n = F.rms_norm(q, (D,)), F.rms_norm(q2, (D,))
            qr, q2r = TT.apply_rotary_emb(qn, cos, sin), TT.apply_rotary_emb(q2n, cos, sin)
            kn, k2n = F.rms_norm(k, (D,)), F.rms_norm(k2, (D,))
            kr, k2r = TT.apply_rotary_emb(kn, cos, sin), TT.apply_rotary_emb(k2n, cos, sin)
            l0, l1 = float(model.transformer.h[8].lambdas[0]), float(model.transformer.h[8].lambdas[1])
            Wk = attn.c_k.weight.detach().float()[D:2 * D, :]; Wk2 = attn.c_k2.weight.detach().float()[D:2 * D, :]   # head 1 rows
            for i, row in enumerate(chunk):
                t, s = row.final, offsets[row.row_id]
                native1 = float(qr[i, t, 1].float() @ kr[i, s, 1].float()) / D
                native2 = float(q2r[i, t, 1].float() @ k2r[i, s, 1].float()) / D
                Cw = v13.writer_contributions(tr[i], s, upto=8)
                live = l0 * sum(Cw.values()) + l1 * tr[i][("embed", s)]
                rho_live = float(live.pow(2).mean().sqrt())
                raw1 = (Wk @ (live / rho_live)); raw2 = (Wk2 @ (live / rho_live))
                rho1, rho2 = float(raw1.pow(2).mean().sqrt()), float(raw2.pow(2).mean().sqrt())
                # rotary at position s is linear: apply it to each writer's key term via a one-hot batch
                def rot(vec, which):
                    full = torch.zeros(1, T, H, D, device=x.device, dtype=torch.float32)
                    full[0, s, 1] = vec
                    return TT.apply_rotary_emb(full, cos[:, :T].float() if cos.dim() == 4 else cos, sin[:, :T].float() if sin.dim() == 4 else sin)[0, s, 1]
                terms1, terms2 = {}, {}
                for w in WRITERS:
                    cw = l0 * Cw[w] + (l1 * tr[i][("embed", s)] if w == "embed" else 0.0)
                    key1 = rot((Wk @ (cw / rho_live)) / rho1, 1); key2 = rot((Wk2 @ (cw / rho_live)) / rho2, 2)
                    terms1[w] = float(qr[i, t, 1].float() @ key1) / D
                    terms2[w] = float(q2r[i, t, 1].float() @ key2) / D
                out[row.row_id] = {"native": (native1, native2), "terms1": terms1, "terms2": terms2}
    return out


def main() -> None:
    disc = L.build_rows()
    if L.rows_sha256(disc) != v1.EXPECTED_ROWS_SHA256:
        raise SystemExit("rows changed")
    nat, sha = v20.load_rows()
    doc = json.loads(v20.ROWS.read_text())
    off_nat = {hashlib.sha256(json.dumps([r["doc_index"], r["position"]]).encode()).hexdigest()[:24]: r["cue_offset"] for r in doc["rows"]}
    off_disc = {r.row_id: r.source_positions[0] - 1 for r in disc}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps({"candidate_id": CANDIDATE_ID, "rows": len(disc) + len(nat), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
                          "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}, indent=2)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    fw = L.ManualForward(backend)
    forwards = 0
    res = {"discovery": key_terms(fw, disc, off_disc), "natural": key_terms(fw, nat, off_nat)}; forwards += 8
    closure = 0.0
    report = {}
    for panel, table in res.items():
        rows = disc if panel == "discovery" else nat
        # keep rows in the top half of native pattern (where 8.1 attends)
        pat = {rid: v["native"][0] * v["native"][1] for rid, v in table.items()}
        cutoff = sorted(pat.values())[len(pat) // 2]
        keep = [rid for rid, v in pat.items() if v >= cutoff]
        sums = {"s1": {w: 0.0 for w in WRITERS}, "s2": {w: 0.0 for w in WRITERS}}
        tot = {"s1": 0.0, "s2": 0.0}
        for rid, v in table.items():
            for key, terms, nat_val in (("s1", v["terms1"], v["native"][0]), ("s2", v["terms2"], v["native"][1])):
                closure = max(closure, abs(sum(terms.values()) - nat_val) / max(abs(nat_val), 1e-6))
                if rid in keep:
                    for w in WRITERS:
                        sums[key][w] += terms[w]
                    tot[key] += nat_val
        shares = {key: {w: sums[key][w] / tot[key] for w in WRITERS} for key in ("s1", "s2")}
        ranked = {key: sorted(WRITERS, key=lambda w: -abs(shares[key][w])) for key in shares}
        report[panel] = {"rows_kept": len(keep), "shares": shares, "ranked": ranked,
                         "top4_share": {key: sum(shares[key][w] for w in ranked[key][:4]) for key in shares}}
        print(panel, {key: [(w, round(shares[key][w], 3)) for w in ranked[key][:5]] for key in shares}, "top4", {k: round(v, 3) for k, v in report[panel]["top4_share"].items()})
    d, n = report["discovery"], report["natural"]
    predictions = {
        "pred_a_key_writer_closure": closure <= CLOSURE_TOL,
        "pred_b_few_writers_build_the_key": all(d["top4_share"][k] >= TOP4_MIN for k in ("s1", "s2")),
        "pred_c_same_writers_on_templates_and_corpus": all(w in d["ranked"][k][:4] for k in ("s1", "s2") for w in n["ranked"][k][:2]),
    }
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    result = {"schema": "aspectual_anchor_dod_cue_key_writers_result_v23", "candidate_id": CANDIDATE_ID, "closure_max_relative_error": closure,
              "panels": report, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "closure": closure, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
