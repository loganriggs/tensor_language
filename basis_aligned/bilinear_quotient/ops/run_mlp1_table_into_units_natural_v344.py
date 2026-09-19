#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_rows_load pred_b_split_closure pred_c_table_part_dominates_on_text_for_3465 pred_d_table_part_dominates_on_text_for_all_four pred_e_units_separate_on_text_nouns
"""On natural text: MLP 1's lookup entry into the agreement units' factors (v344). v341 / v343: on the panel rows 67-88% of MLP 1's contribution to each factor of
3465 / 493 / 1036 / 829 is its context-free entry. OOD on the 128 natural verb rows (v272 FineWeb + v273 Pile; the cue noun at position cue_offset, plural or
singular by the miner's label): the same exact split at the cue noun, contrast = plural-cue rows minus singular-cue rows (pooled, unpaired), plus each unit's
plural / singular separation at the cue in pooled std.
PREDICTIONS (scored as written; failures preserved; priors from v343)
    pred_a_rows_load                            128 rows load with a valid cue position and both cue classes present (instrument)
    pred_b_split_closure                        table part + remainder part = MLP 1's term within relative 1e-3 on every row, unit and factor
    pred_c_table_part_dominates_on_text_for_3465  for 3465 the table share of MLP 1's pooled contrast is >= 0.60 on both factors
    pred_d_table_part_dominates_on_text_for_all_four  the same for 493, 1036 and 829. Prior: unsure -- real contexts vary the remainder far more than the panel's "The".
    pred_e_units_separate_on_text_nouns         at the cue noun each of the four units separates plural from singular cue rows by >= 1 pooled std with its panel sign
PRICE (registered maximum): 2 natural batches + 1 table batch (unique cue nouns) = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_context_gain_decomposition_v289 as v289

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/mlp1_table_into_units_natural_v344_result.json"
CANDIDATE_ID = "pronoun_number.mlp1_table_into_units_natural_v344"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829,)}; BATCH = 64
NATURAL = [ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v272.json", ROOT / "circuits/followups/pronoun_number_dod_natural_verb_rows_v273.json"]
SPLIT_TOL, TABLE_MIN, GAP_STD = 1e-3, 0.60, 1.0
PANEL_SIGN = {3465: -1, 493: 1, 1036: 1, 829: 1}
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_rows_load": "128 rows, both classes", "pred_b_split_closure": "<= 1e-3", "pred_c_table_part_dominates_on_text_for_3465": ">= 0.60 x 2", "pred_d_table_part_dominates_on_text_for_all_four": ">= 0.60 x 6", "pred_e_units_separate_on_text_nouns": ">= 1 std, panel sign x 4"}


def main() -> None:
    recs = [r for p in NATURAL for r in json.loads(p.read_text())["rows"]]
    cue = [int(r["cue_offset"]) for r in recs]; plural = [i for i, r in enumerate(recs) if r["cue"] == "plural"]; sing = [i for i, r in enumerate(recs) if r["cue"] != "plural"]
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(recs), "plural_rows": len(plural), "singular_rows": len(sing), "units": {str(k): list(v) for k, v in UNITS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"split_tol": SPLIT_TOL, "table_min": TABLE_MIN, "gap_std": GAP_STD}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0; blocks = model.transformer.h
    lam = [(float(b.lambdas[0]), float(b.lambdas[1])) for b in blocks]; nat = torch.tensor([r["ids"] for r in recs], device="cuda")
    W1 = {l: [] for l in UNITS}; XIN = {l: [] for l in UNITS}; H = {u: [] for l in UNITS for u in UNITS[l]}
    with torch.no_grad():
        for s0 in range(0, len(recs), BATCH):
            chunk = nat[s0:s0 + BATCH]; idx = torch.arange(chunk.shape[0]); pn = torch.tensor(cue[s0:s0 + BATCH])
            x = F.rms_norm(model.transformer.wte(chunk), (model.config.n_embd,)); x0, v1_ = x, None; m1 = None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in UNITS:
                    XIN[l].append(x[idx, pn].float().cpu()); W1[l].append(m1[idx, pn].float().cpu()); hh = dod_units.hidden(model, block.mlp, xin).float().cpu()
                    for u in UNITS[l]: H[u].append(hh[idx, pn, u])
                m = block.mlp(xin); x = x + m
                if l == 1: m1 = m
                if l == 8: break
            forwards += 1
    cue_ids = sorted({int(recs[i]["ids"][cue[i]]) for i in range(len(recs))}); tok = torch.tensor(cue_ids, device="cuda"); tab = v289.capture(backend, tok.unsqueeze(1), torch.zeros(len(cue_ids), dtype=torch.long, device="cuda")); forwards += 1
    tindex = {t: i for i, t in enumerate(cue_ids)}; T = torch.stack([tab["mlp1"][tindex[int(recs[i]["ids"][cue[i]])]] for i in range(len(recs))])
    per, sclos, seps = {}, 0.0, {}
    for l in UNITS:
        scale = 1.0
        for j in range(2, l + 1): scale *= lam[j][0]
        Xl = torch.cat(XIN[l]); rms = Xl.pow(2).mean(1).sqrt(); Wm = torch.cat(W1[l]); alpha = (Wm * (scale * T)).sum(1) / ((scale * T) ** 2).sum(1); A = alpha[:, None] * scale * T; Rm = Wm - A
        for u in UNITS[l]:
            h = torch.cat(H[u]); p_, s_ = h[plural], h[sing]; std = float(torch.cat([p_ - p_.mean(), s_ - s_.mean()]).std()); seps[str(u)] = float((p_.median() - s_.median()) / std)
            Lr, Rr = blocks[l].mlp.Left.weight.detach().float()[u].cpu(), blocks[l].mlp.Right.weight.detach().float()[u].cpu()
            for name, vec in (("L", Lr), ("R", Rr)):
                term = lambda M: float(((M @ vec)[plural] / rms[plural]).mean() - ((M @ vec)[sing] / rms[sing]).mean())
                tot, ta, tr = term(Wm), term(A), term(Rm); sclos = max(sclos, abs(ta + tr - tot) / max(abs(tot), 1e-6))
                per[f"mlp{l}.{u}.{name}"] = {"mlp1_contrast_term": tot, "table_share": ta / tot, "remainder_share": tr / tot}
    report = {"split_closure": sclos, "per_unit_factor": per, "separation_at_cue_over_std": seps, "alpha_median_at_block3": float(alpha.median())}
    print(json.dumps(report, indent=1))
    ts = lambda u, n: per[[k for k in per if k.split(".")[1] == str(u)][0].rsplit(".", 1)[0] + f".{n}"]["table_share"]
    predictions = {"pred_a_rows_load": len(recs) == 128 and len(plural) > 0 and len(sing) > 0, "pred_b_split_closure": sclos <= SPLIT_TOL, "pred_c_table_part_dominates_on_text_for_3465": all(ts(3465, n) >= TABLE_MIN for n in ("L", "R")),
                   "pred_d_table_part_dominates_on_text_for_all_four": all(ts(u, n) >= TABLE_MIN for u in (493, 1036, 829) for n in ("L", "R")), "pred_e_units_separate_on_text_nouns": all(seps[str(u)] * PANEL_SIGN[u] >= GAP_STD for u in PANEL_SIGN)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "mlp1_table_into_units_natural_result_v344", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
