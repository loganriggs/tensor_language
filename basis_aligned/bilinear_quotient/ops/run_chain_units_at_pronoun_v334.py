#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_noun_position_separates pred_c_final_position_separates_for_1036_829 pred_d_3465_493_silent_at_final pred_e_noun_gap_replays_single_token_order
"""Do the chain's named units read the pronoun itself? (v334). v332: from the token alone, MLP-5 unit 1036 and MLP-8 unit 829 give plural pronouns half the
singular -> plural axis, while MLP-3 units 3465 / 493 give them nothing. On the v76 pronoun rows the pronoun is the answer at the final position, so the
question is what these units do at the FINAL position natively (where the model predicts they vs he): the rows differ only in the noun's number, so a
unit that separates plural from singular rows at the final position is carrying the noun's number there (the copy), not reading a pronoun token. Read
h_j at the noun position and at the final position, plural vs singular rows, for 3465, 493, 1036, 829, 953, 1030; gap in pooled std and the fraction of
aligned pairs with the sign of the pooled gap. Native forward only.
PREDICTIONS (scored as written; failures preserved; priors from v168 / v194 / v296 / v332)
    pred_a_closure                              the manual forward reproduces the model's own they - he margin (2.048) within 1e-3 (instrument)
    pred_b_noun_position_separates              at the noun, every one of the six units separates plural from singular rows by >= 1 pooled std
    pred_c_final_position_separates_for_1036_829  at the final position, 1036 and 829 separate the rows by >= 1 pooled std (the number state reaches the answer position through them)
    pred_d_3465_493_silent_at_final             at the final position, 3465 and 493 separate by < 0.5 pooled std (lexical detectors of the noun, not carriers to the answer). Prior: unsure.
    pred_e_noun_gap_replays_single_token_order  the ranking of the six units by |gap| at the noun shares its top-3 with v332's single-token ranking (1036, 829, 3465)
PRICE (registered maximum): 3 row batches x 1 forward = 3 forwards; 0 backwards; 0 fits. Bar <= 5.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_pronoun_number_dod_battery_v76 as g
import dod_battery, dod_units

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/chain_units_at_pronoun_v334_result.json"
CANDIDATE_ID = "pronoun_number.chain_units_at_pronoun_v334"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; BATCH = 32
CLOSURE_TOL, NATIVE_M, GAP_STD, SILENT = 1e-3, 2.0481, 1.0, 0.5
FORWARDS_MAX = 5
PREDICTIONS = {"pred_a_closure": "margin 2.048 +- 1e-3", "pred_b_noun_position_separates": ">= 1 std x 6", "pred_c_final_position_separates_for_1036_829": ">= 1 std x 2", "pred_d_3465_493_silent_at_final": "< 0.5 std x 2", "pred_e_noun_gap_replays_single_token_order": "top-3 shared"}


def main() -> None:
    rows, he, she, agents, objects = g.build()
    nouns = {L._single(" " + a) for a in agents} | {L._single(" " + a + "s") for a in agents}
    noun_of = lambda row: next(i for i, t in enumerate(row.ids) if t in nouns)
    plan = {"candidate_id": CANDIDATE_ID, "rows": len(rows), "rows_sha256": L.rows_sha256(rows), "units": {str(k): list(v) for k, v in UNITS.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "native_m": NATIVE_M, "gap_std": GAP_STD, "silent": SILENT}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; fw = L.ManualForward(backend); forwards = 0; blocks = model.transformer.h
    H = {"noun": {k: [] for l in UNITS for k in UNITS[l]}, "final": {k: [] for l in UNITS for k in UNITS[l]}}; margins = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]; tokens = fw._tokens(chunk); idx = torch.arange(len(chunk)); pn = torch.tensor([noun_of(r) for r in chunk]); pf = torch.tensor([r.final for r in chunk])
            x = F.rms_norm(model.transformer.wte(tokens), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in UNITS:
                    hh = dod_units.hidden(model, block.mlp, xin).float().cpu()
                    for u in UNITS[l]: H["noun"][u].append(hh[idx, pn, u]); H["final"][u].append(hh[idx, pf, u])
                x = x + block.mlp(xin)
            logits = 30 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30)
            margins += [float(logits[i, r.final, L._single(" they")] - logits[i, r.final, L._single(" he")]) for i, r in enumerate(chunk)]
            forwards += 1
    base = sum((m if r.present else -m) for m, r in zip(margins, rows)) / len(rows); closure = abs(base - NATIVE_M)
    partner = {(row.construction, row.group, row.present): i for i, row in enumerate(rows)}; plural = [i for i, r in enumerate(rows) if r.present]
    per = {}
    for pos in ("noun", "final"):
        for u in H[pos]:
            h = torch.cat(H[pos][u]); p_ = h[plural]; s_ = h[[partner[(rows[i].construction, rows[i].group, False)] for i in plural]]
            gap = float(p_.median() - s_.median()); std = float(torch.cat([p_ - p_.mean(), s_ - s_.mean()]).std()); d = p_ - s_
            per[f"{pos}.{u}"] = {"gap_over_std": gap / std, "plural_median": float(p_.median()), "singular_median": float(s_.median()), "pair_sign_fraction": float(((d * gap) > 0).float().mean())}
    report = {"closure_margin_gap": closure, "native_margin": base, "per": per}
    print(json.dumps(report, indent=1))
    gn = lambda u: abs(per[f"noun.{u}"]["gap_over_std"]); gf = lambda u: abs(per[f"final.{u}"]["gap_over_std"])
    top3 = set(sorted([u for l in UNITS for u in UNITS[l]], key=lambda u: -gn(u))[:3])
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_noun_position_separates": all(gn(u) >= GAP_STD for l in UNITS for u in UNITS[l]), "pred_c_final_position_separates_for_1036_829": gf(1036) >= GAP_STD and gf(829) >= GAP_STD,
                   "pred_d_3465_493_silent_at_final": gf(3465) < SILENT and gf(493) < SILENT, "pred_e_noun_gap_replays_single_token_order": top3 == {1036, 829, 3465}}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "chain_units_at_pronoun_result_v334", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
