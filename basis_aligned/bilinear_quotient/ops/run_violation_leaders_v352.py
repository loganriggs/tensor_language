#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_leaders_flag_the_violation_cell pred_c_leaders_are_interaction_units pred_d_leaders_mirror_cell pred_e_leaders_silent_alone
"""The leading violation units of MLP 3: 3040, 114, 565 (v352). v351: at the verb of "The X are" (X singular) these three units carry the largest
write-weighted contrast against "The X is", and zeroing the top ten costs 0.094 nats and 0.73 log-odds of plural continuation. Here their full 2 x 2 table
on the "The X is / are" frames (verb-form effect, noun-number effect, agreement interaction, in pooled std) and their single-token behaviour (the 256 pairs'
singular and plural nouns alone; is / are alone) to say whether they are agreement-interaction units (fire for the mismatch specifically), verb-form or
noun-number readers that happen to differ across cells, or lexical units.
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_closure                        the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_leaders_flag_the_violation_cell for each of 3040 / 114 / 565, the "singular X + are" cell differs from the "singular X + is" cell by >= 2 pooled std
    pred_c_leaders_are_interaction_units  for each of the three, |agreement interaction| >= |verb-form effect| and >= |noun-number effect| (in std)
    pred_d_leaders_mirror_cell            for each of the three, the "plural X + is" cell also differs from "plural X + are" by >= 1 pooled std with the same sign as the singular violation (a symmetric mismatch detector). Prior: unsure.
    pred_e_leaders_silent_alone           from single tokens alone, each of the three separates plural from singular nouns by < 1 pooled std and is / are by < 1 pooled std (they need both tokens)
PRICE (registered maximum): 4 cells x 256 rows / 256 = 4 forwards + 1 single-token batch (514 tokens, 3 batches) = 7 forwards (blocks 0-3); 0 backwards; 0 fits. Bar <= 9.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/violation_leaders_v352_result.json"
CANDIDATE_ID = "pronoun_number.violation_leaders_v352"
UNITS = {3: (3040, 114, 565, 3465)}; TOP = 3
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, VIOL_MIN, MIRROR_MIN, ALONE_MAX = 1e-4, 2.0, 1.0, 1.0
VERBS = (" is", " are"); LEADERS = ("mlp3.3040", "mlp3.114", "mlp3.565")
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 9
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_leaders_flag_the_violation_cell": ">= 2 std x 3", "pred_c_leaders_are_interaction_units": "|inter| >= |form|, |noun| x 3", "pred_d_leaders_mirror_cell": ">= 1 std same sign x 3", "pred_e_leaders_silent_alone": "< 1 std alone x 3"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "viol_min": VIOL_MIN, "mirror_min": MIRROR_MIN, "alone_max": ALONE_MAX}, "verbs": list(VERBS), "frame": "The X verb"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    blocks = model.transformer.h
    cells = {}; closure = 0.0
    for vb in VERBS:
        vid = L._single(vb); H, H2 = {}, {}
        with torch.no_grad():
            for s0 in range(0, len(tokens), 256):
                the = L._single("The"); ids = torch.tensor([[the, t, vid] for t in tokens[s0:s0 + 256]], device="cuda")
                x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if l in UNITS:
                        hh = dod_units.hidden(model, block.mlp, xin)[:, 2].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, 2].float().cpu()
                        for u in UNITS[l]: H.setdefault((l, u), []).append(hh[:, u]); H2.setdefault((l, u), []).append(hh2[:, u])
                    if l == TOP: break
                    x = x + block.mlp(xin)
                forwards += 1
        H = {kk: torch.cat(v) for kk, v in H.items()}; H2 = {kk: torch.cat(v) for kk, v in H2.items()}
        closure = max(closure, max(float(((H2[kk] - H[kk]).abs() / H[kk].abs().clamp_min(1e-6)).max()) for kk in H)); tindex = {t: i for i, t in enumerate(tokens)}
        for (l, u), h in H.items():
            cells[(f"mlp{l}.{u}", vb, "s")] = torch.tensor([float(h[tindex[a_]]) for a_, _ in pairs]); cells[(f"mlp{l}.{u}", vb, "p")] = torch.tensor([float(h[tindex[b_]]) for _, b_ in pairs])
    per = {}
    for key in sorted({k[0] for k in cells}):
        c = {(vb, n): cells[(key, vb, n)] for vb in VERBS for n in ("s", "p")}; allv = torch.cat(list(c.values())); std = float(torch.cat([v - v.mean() for v in c.values()]).std())
        med = {f"{vb.strip()}|{n}": float(v.median()) for (vb, n), v in c.items()}
        form = ((med["are|s"] + med["are|p"]) - (med["is|s"] + med["is|p"])) / 2 / std; noun = ((med["is|p"] + med["are|p"]) - (med["is|s"] + med["are|s"])) / 2 / std
        gram = (med["is|s"] + med["are|p"]) / 2; ungram = (med["is|p"] + med["are|s"]) / 2; inter = (gram - ungram) / std
        per[key] = {"medians": med, "pooled_std": std, "verb_form_effect_std": form, "noun_number_effect_std": noun, "agreement_interaction_std": inter}
    report = {"closure_max": closure, "per_unit": per}
    print(json.dumps(report, indent=1))
    # single tokens alone: the 512 nouns and is / are
    alone_tokens = tokens + [L._single(" is"), L._single(" are")]; HA = {}
    with torch.no_grad():
        for s0 in range(0, len(alone_tokens), 256):
            ids = torch.tensor(alone_tokens[s0:s0 + 256], device="cuda").unsqueeze(1); x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l == 3:
                    hh = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu()
                    for u in UNITS[3]: HA.setdefault(u, []).append(hh[:, u])
                    break
                x = x + block.mlp(xin)
            forwards += 1
    HA = {u: torch.cat(v) for u, v in HA.items()}; ai = {t: i for i, t in enumerate(alone_tokens)}
    alone = {}
    for u, h in HA.items():
        s_ = torch.tensor([float(h[ai[a_]]) for a_, _ in pairs]); p_ = torch.tensor([float(h[ai[b_]]) for _, b_ in pairs]); std = float(torch.cat([s_ - s_.mean(), p_ - p_.mean()]).std())
        alone[f"mlp3.{u}"] = {"noun_sep_std": float((p_.median() - s_.median()) / std), "are_minus_is_over_std": float((h[ai[L._single(" are")]] - h[ai[L._single(" is")]]) / std)}
    report["alone"] = alone
    for k in LEADERS:
        m = per[k]["medians"]; sd = per[k]["pooled_std"]; per[k]["viol_gap_std"] = (m["are|s"] - m["is|s"]) / sd; per[k]["mirror_gap_std"] = (m["is|p"] - m["are|p"]) / sd
    print(json.dumps({k: {a_: (round(b_, 2) if isinstance(b_, float) else b_) for a_, b_ in per[k].items() if a_ != "medians"} for k in LEADERS}, indent=1)); print(json.dumps(alone, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_leaders_flag_the_violation_cell": all(abs(per[k]["viol_gap_std"]) >= VIOL_MIN for k in LEADERS),
                   "pred_c_leaders_are_interaction_units": all(abs(per[k]["agreement_interaction_std"]) >= max(abs(per[k]["verb_form_effect_std"]), abs(per[k]["noun_number_effect_std"])) for k in LEADERS),
                   "pred_d_leaders_mirror_cell": all(abs(per[k]["mirror_gap_std"]) >= MIRROR_MIN and per[k]["mirror_gap_std"] * per[k]["viol_gap_std"] > 0 for k in LEADERS),
                   "pred_e_leaders_silent_alone": all(abs(alone[k]["noun_sep_std"]) < ALONE_MAX and abs(alone[k]["are_minus_is_over_std"]) < ALONE_MAX for k in LEADERS)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "violation_leaders_result_v352", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
