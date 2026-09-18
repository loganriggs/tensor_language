#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_fold_closure pred_b_head_11_3_largest_source_is_the_head_noun pred_c_relay_heads_write_11_3_source pred_d_source_state_is_mostly_mlp_written pred_e_token_only_share_at_most_030
"""Perfect have/has DoD (v103): SOURCE fold of the four readout coefficients + WRITER fold of the state 11.3 reads (better_circuits §3.8).

Lane: Claude circuit lane. Parents: v97 (set {11.3, 7.8, 5.3, 9.7}, 66% fresh), v99 (60% direct, MLP suffix relay), v102 (pair terms
with 11.3 carry 82% of the interaction: a serial relay into 11.3). Part 1 (v80's exact fold): each head's have−has coefficient at the final
query split by source category -- head noun, determiner before it, prefix word(s), preposition, determiner before the object, object /
final -- and by value branch. Part 2 (v82's exact writer fold at block 11): at the position that carries most of 11.3's contrast, the
state 11.3 reads (r = V_{11.3}^T v_hat) decomposed into embedding, every head of blocks 0-10, c_proj biases and MLPs 0-10. If 7.8 / 9.7 /
5.3 appear among the writers of that state, the relay of v102 is a mechanism, not a residue. Evidence tag: fold, fresh rows.

PREDICTIONS (scored as written; failures preserved)
    pred_a_fold_closure                              both folds close within relative 1e-3 (part 1: sum of terms = captured v . w; part 2: recon of live_11)
    pred_b_head_11_3_largest_source_is_the_head_noun for 11.3 the head-noun position carries the largest |share| of its contrast. Prior: unsure (the
                                                     number may sit at the object / final token).
    pred_c_relay_heads_write_11_3_source             at 11.3's largest source position, attnhead 7.8 + 9.7 + 5.3 together carry >= 0.20 of the
                                                     reader-projected writer contrast. Prior: unsure.
    pred_d_source_state_is_mostly_mlp_written        at that position the MLP writers carry >= 0.50 (v82's shape on the pronoun lines). Prior: unsure.
    pred_e_token_only_share_at_most_030              pooled over the four heads the block-0 value (token-only) branch is <= 0.30 of the contrast

PRICE (registered maximum): 3 batches x (capture + 4 block folds) + 3 batches x (positional trace + block-11 fold) = 21 forwards; 0 backwards;
0 fits. Bar <= 24.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_aspectual_dod_removal_v1 as v1
import run_perfect_number_dod_battery_v97 as line
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/perfect_number_dod_relay_fold_v103_result.json"
CANDIDATE_ID = "perfect_number.have_vs_has.dod_relay_fold_v103"
CLOSURE_TOL, RELAY_MIN, MLP_MIN, INHERITED_MAX = 1e-3, 0.20, 0.50, 0.30
FORWARDS_MAX = 24
LAYER, HEAD = 11, 3
CATEGORIES = ("prefix", "det_noun", "noun", "prep", "det_object", "final")
PREDICTIONS = {"pred_a_fold_closure": "<= 1e-3", "pred_b_head_11_3_largest_source_is_the_head_noun": "noun", "pred_c_relay_heads_write_11_3_source": ">= 0.20",
               "pred_d_source_state_is_mostly_mlp_written": ">= 0.50", "pred_e_token_only_share_at_most_030": "<= 0.30"}
PREPS = {L._single(" by"), L._single(" inside"), L._single(" behind")}
DET = {L._single(" the"), L._single("the")}
WRITERS = ["embed"] + [f"attnhead:{l:02d}:{h}" for l in range(LAYER) for h in range(9)] + [f"attnbias:{l:02d}" for l in range(LAYER)] + [f"mlp:{l:02d}" for l in range(LAYER)]


def positions(row, nouns):
    ids = row.ids
    noun = next(i for i, t in enumerate(ids) if t in nouns); prep = next(i for i, t in enumerate(ids) if t in PREPS)
    dets = [i for i, t in enumerate(ids) if t in DET]
    return {"noun": noun, "prep": prep, "det_noun": max(i for i in dets if i < noun), "det_object": max(i for i in dets if i > prep), "final": row.final}


def category(row, s, nouns):
    P = positions(row, nouns)
    for k, v in P.items():
        if s == v:
            return k
    return "prefix"


def contributions(tr, pos, upto):
    x0 = tr[("embed", pos)]; C = {"embed": x0.clone()}
    for l in range(upto):
        l0, l1 = tr[f"lambda0:{l:02d}"], tr[f"lambda1:{l:02d}"]
        for k in C:
            C[k] = l0 * C[k]
        C["embed"] = C["embed"] + l1 * x0
        heads = {f"attnhead:{l:02d}:{h}": tr[(f"attnhead:{l:02d}:{h}", pos)].clone() for h in range(9)}
        C.update(heads); C[f"attnbias:{l:02d}"] = tr[(f"attn:{l:02d}", pos)] - sum(heads.values()); C[f"mlp:{l:02d}"] = tr[(f"mlp:{l:02d}", pos)].clone()
    return C


def main() -> None:
    rows, have, has, agents, objects = line.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "heads": list(line.HEADS), "categories": CATEGORIES, "writers": len(WRITERS), "forwards_max": FORWARDS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"closure_tol": CLOSURE_TOL, "relay_min": RELAY_MIN, "mlp_min": MLP_MIN, "inherited_max": INHERITED_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); model = backend.model
    fw = L.ManualForward(backend)
    comps = dod_battery.LineSpec(CANDIDATE_ID, OUT.name, rows, have, has, line.HEADS).set_components()
    fw.directions = L.readout_directions(model, comps, have, has)
    forwards, store, terms = 0, {}, [dict() for _ in rows]
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        store.update(fw.capture(chunk, comps)); forwards += 1
        for comp in comps:
            out, lamb = L.head_source_terms(fw, chunk, comp, fw.directions); forwards += 1
            for j, entry in enumerate(out):
                for head, val in entry.items():
                    terms[start + j][f"{comp.layer}.{head}"] = (comp, head, val)
    closure1 = 0.0
    for row, t in zip(rows, terms):
        for key, (comp, head, val) in t.items():
            w = store[(row.row_id, comp.name, row.final, head)].float(); v = fw.directions[(comp.name, head)].float(); v = v / v.norm()
            direct = float(w @ v.to(w.device)); closure1 = max(closure1, abs(val["coefficient"] - direct) / max(abs(direct), 1e-6))
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}
    part1 = {}
    for key in sorted({k for t in terms for k in t}):
        pooled = {c: {"current": 0.0, "inherited": 0.0} for c in CATEGORIES}; contrast, n = 0.0, 0
        for i, row in enumerate(rows):
            if not row.present:
                continue
            j = partner[(row.construction, row.group, False)]; a, b = terms[i][key][2], terms[j][key][2]
            if len(a["pattern"]) != len(b["pattern"]):
                raise SystemExit("pair positions are not aligned")
            for s in range(len(a["pattern"])):
                c = category(row, s, nouns); pooled[c]["current"] += a["term_current"][s] - b["term_current"][s]; pooled[c]["inherited"] += a["term_inherited"][s] - b["term_inherited"][s]
            contrast += a["coefficient"] - b["coefficient"]; n += 1
        shares = {c: (pooled[c]["current"] + pooled[c]["inherited"]) / contrast for c in CATEGORIES}
        part1[key] = {"mean_contrast": contrast / n, "shares": shares, "inherited_share": sum(pooled[c]["inherited"] for c in CATEGORIES) / contrast, "largest_source": max(CATEGORIES, key=lambda c: abs(shares[c])), "by_category_branch": pooled}
        print(key, "contrast", round(contrast / n, 3), {c: round(s, 3) for c, s in shares.items()}, "inherited", round(part1[key]["inherited_share"], 3))
    tot = sum(v["mean_contrast"] for v in part1.values()); pooled_inh = sum(v["inherited_share"] * v["mean_contrast"] for v in part1.values()) / tot
    top_pos = part1["11.3"]["largest_source"]
    # part 2: writer fold of the state 11.3 reads at its top position
    comp11 = next(c for c in comps if c.layer == LAYER)
    r = L.reader_directions(model, comp11, fw.directions)[HEAD]
    block = model.transformer.h[LAYER]; l0_L, l1_L = float(block.lambdas[0]), float(block.lambdas[1])
    traces = []
    for start in range(0, len(rows), v1.BATCH):
        chunk = rows[start:start + v1.BATCH]
        traces.extend(L.forward_trace_positions(fw, chunk, lambda rw: [positions(rw, nouns)[top_pos]], upto_layer=LAYER, head_write_layers=range(LAYER))); forwards += 1
    closure2, per_row = 0.0, []
    for i, (row, tr) in enumerate(zip(rows, traces)):
        pos = positions(row, nouns)[top_pos]
        C = contributions(tr, pos, LAYER); C = {k: l0_L * v for k, v in C.items()}; C["embed"] = C["embed"] + l1_L * tr[("embed", pos)]
        true = tr[("live", pos)]; recon = sum(C.values()); closure2 = max(closure2, float((recon - true).norm() / true.norm()))
        rms = float(true.pow(2).mean().sqrt()); scale = terms[i]["11.3"][2]["pattern"][pos] * (1 - float(block.attn.lamb)) / rms
        rr = r.to(true.device); per_row.append({w: float(rr @ C[w]) * scale for w in WRITERS})
    totals = {w: 0.0 for w in WRITERS}; contrast = 0.0
    for i, row in enumerate(rows):
        if not row.present:
            continue
        j = partner[(row.construction, row.group, False)]
        for w in WRITERS:
            totals[w] += per_row[i][w] - per_row[j][w]
        contrast += sum(per_row[i].values()) - sum(per_row[j].values())
    shares = {w: totals[w] / contrast for w in WRITERS}
    heads = {w: s for w, s in shares.items() if w.startswith("attnhead")}; ranked = sorted(heads, key=lambda w: -abs(heads[w]))
    relay = sum(shares[f"attnhead:{l:02d}:{h}"] for l, h in ((7, 8), (9, 7), (5, 3)))
    mlp = sum(s for w, s in shares.items() if w.startswith("mlp"))
    part2 = {"position": top_pos, "contrast_total": contrast, "head_part": sum(heads.values()), "mlp_part": mlp, "embed_share": shares["embed"], "relay_heads_share": relay,
             "relay_heads": {f"{l}.{h}": shares[f"attnhead:{l:02d}:{h}"] for l, h in ((7, 8), (9, 7), (5, 3))}, "top_heads": [(w, heads[w]) for w in ranked[:8]],
             "mlp_by_block": {w: s for w, s in shares.items() if w.startswith("mlp")}}
    print("11.3 reads at", top_pos, "heads", round(part2["head_part"], 3), "mlp", round(mlp, 3), "embed", round(shares["embed"], 3), "relay 7.8/9.7/5.3", {k: round(v, 3) for k, v in part2["relay_heads"].items()}, "top", [(w, round(s, 3)) for w, s in part2["top_heads"][:5]])
    predictions = {"pred_a_fold_closure": max(closure1, closure2) <= CLOSURE_TOL, "pred_b_head_11_3_largest_source_is_the_head_noun": top_pos == "noun",
                   "pred_c_relay_heads_write_11_3_source": relay >= RELAY_MIN, "pred_d_source_state_is_mostly_mlp_written": mlp >= MLP_MIN, "pred_e_token_only_share_at_most_030": pooled_inh <= INHERITED_MAX}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "perfect_number_dod_relay_fold_result_v103", "candidate_id": CANDIDATE_ID, "plan": plan, "closure": {"part1": closure1, "part2": closure2}, "part1": part1, "pooled_inherited_share": pooled_inh,
                               "part2": part2, "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
