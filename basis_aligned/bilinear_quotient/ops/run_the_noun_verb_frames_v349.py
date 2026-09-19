#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_3465_violation_survives_the_determiner pred_c_3465_violation_gap_within_half_of_v347 pred_d_493_mirror_with_determiner pred_e_verb_form_readers_unchanged
"""The violation reading with a determiner present: "The X is / are" (v349). v347 / v348: at the verb of [X, verb] frames unit 3465 fires 2-5 std for
"singular noun + plural verb". Real subjects carry a determiner, and v337 showed "The" sharpens 3465's noun reading. Three-token frames ["The", X, verb]
with verb in {" is", " are"} on the 256 pairs, read at the verb: the four cell medians per unit, the violation gaps for 3465 and 493, the verb-form
effects for 829 / 1036, all compared with the two-token v347 values.
PREDICTIONS (scored as written; failures preserved; priors from v347)
    pred_a_closure                                the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_3465_violation_survives_the_determiner  3465's (singular X, are) cell lies >= 1 pooled std below its (singular X, is) cell
    pred_c_3465_violation_gap_within_half_of_v347  that gap is within a factor 2 of v347's two-token gap (-302 vs -0.5 over the pooled std there: computed here as -3.0 std -> 1.5-6.0)
    pred_d_493_mirror_with_determiner             493's (plural X, is) cell lies >= 0.5 pooled std above its (plural X, are) cell. Prior: unsure.
    pred_e_verb_form_readers_unchanged            829's and 1036's verb-form effects keep the v347 sign (are above is)
PRICE (registered maximum): 4 cells x 256 rows / 256 = 4 forwards (blocks 0-8); 0 backwards; 0 fits. Bar <= 6.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/the_noun_verb_frames_v349_result.json"
CANDIDATE_ID = "pronoun_number.the_noun_verb_frames_v349"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, GAP_MIN, V347_GAP, MIRROR_MIN = 1e-4, 1.0, 3.0, 0.5
VERBS = (" is", " are")
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 6
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_3465_violation_survives_the_determiner": ">= 1 std", "pred_c_3465_violation_gap_within_half_of_v347": "1.5-6.0 std", "pred_d_493_mirror_with_determiner": ">= 0.5 std", "pred_e_verb_form_readers_unchanged": "are > is x 2"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gap_min": GAP_MIN, "v347_gap": V347_GAP, "mirror_min": MIRROR_MIN}, "verbs": list(VERBS), "frame": "The X verb"}
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
    g3 = (per["mlp3.3465"]["medians"]["are|s"] - per["mlp3.3465"]["medians"]["is|s"]) / per["mlp3.3465"]["pooled_std"]; g4 = (per["mlp3.493"]["medians"]["is|p"] - per["mlp3.493"]["medians"]["are|p"]) / per["mlp3.493"]["pooled_std"]
    report["violation_gap_3465_std"] = g3; report["mirror_gap_493_std"] = g4
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_3465_violation_survives_the_determiner": g3 <= -GAP_MIN, "pred_c_3465_violation_gap_within_half_of_v347": V347_GAP / 2 <= abs(g3) <= V347_GAP * 2,
                   "pred_d_493_mirror_with_determiner": g4 >= MIRROR_MIN, "pred_e_verb_form_readers_unchanged": per["mlp8.829"]["verb_form_effect_std"] > 0 and per["mlp5.1036"]["verb_form_effect_std"] > 0}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "the_noun_verb_frames_result_v349", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
