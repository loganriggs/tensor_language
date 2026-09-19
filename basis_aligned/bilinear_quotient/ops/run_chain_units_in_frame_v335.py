#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_context_sharpens_3465 pred_c_context_sharpens_every_surviving_unit pred_d_953_1030_stay_silent pred_e_plural_side_fraction_rises
"""Does one context token sharpen the vocabulary separation as it did the lexicon's? (v335). v333: from the token alone, 3465 / 493 / 1036 / 829 separate
the 256 vocabulary s-pairs at 2-3 pooled std; v334: on the pronoun rows (lexicon nouns after "The"), 3465's gap is -7.3 std in context vs -4.2 alone. Here the
same 256 vocabulary pairs are read at the noun of the two-token frame ["The", X] (each form alone after "The"), each unit's plural - singular gap in pooled std
and plural-side fraction at 256 pairs, compared with v333's single-token values.
PREDICTIONS (scored as written; failures preserved; priors from v333 / v334)
    pred_a_closure                            the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_context_sharpens_3465              |gap(3465, "The" frame)| >= 1.3 x |gap(3465, alone)| = 1.3 x 2.58
    pred_c_context_sharpens_every_surviving_unit  the same >= 1.3x for 493, 1036 and 829 (v333: 2.65, 2.01, 1.76). Prior: unsure.
    pred_d_953_1030_stay_silent               953 and 1030 remain below 1 pooled std in the frame (context does not create a lexical separation they lack)
    pred_e_plural_side_fraction_rises         the plural-side fraction at 256 pairs rises for each of 3465 / 493 / 1036 / 829 relative to v333 (0.96 / 0.98 / 0.89 / 0.82)
PRICE (registered maximum): 512 two-token rows in 2 batches of 256 (blocks 0-8) = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/chain_units_in_frame_v335_result.json"
CANDIDATE_ID = "pronoun_number.chain_units_in_frame_v335"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, SHARPEN, SILENT, SIZES = 1e-4, 1.3, 1.0, (256,)
V333 = {"mlp3.3465": -2.58, "mlp3.493": 2.65, "mlp5.1036": 2.01, "mlp8.829": 1.76, "mlp8.953": -0.06, "mlp8.1030": -0.04}
V333_SIDE = {"mlp3.3465": 0.96, "mlp3.493": 0.98, "mlp5.1036": 0.89, "mlp8.829": 0.82}
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_context_sharpens_3465": ">= 1.3x", "pred_c_context_sharpens_every_surviving_unit": ">= 1.3x x 3", "pred_d_953_1030_stay_silent": "< 1 std x 2", "pred_e_plural_side_fraction_rises": "rises x 4"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "sharpen": SHARPEN, "silent": SILENT}, "frame": "The X"}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    blocks = model.transformer.h
    H, H2 = {}, {}
    with torch.no_grad():
        for s0 in range(0, len(tokens), 256):
            the = L._single("The"); ids = torch.tensor([[the, t] for t in tokens[s0:s0 + 256]], device="cuda")
            x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in UNITS:
                    hh = dod_units.hidden(model, block.mlp, xin)[:, 1].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, 1].float().cpu()
                    for u in UNITS[l]: H.setdefault((l, u), []).append(hh[:, u]); H2.setdefault((l, u), []).append(hh2[:, u])
                if l == TOP: break
                x = x + block.mlp(xin)
            forwards += 1
    H = {k: torch.cat(v) for k, v in H.items()}; H2 = {k: torch.cat(v) for k, v in H2.items()}
    closure = max(float(((H2[k] - H[k]).abs() / H[k].abs().clamp_min(1e-6)).max()) for k in H); tindex = {t: i for i, t in enumerate(tokens)}
    per = {}
    for (l, u), h in H.items():
        key = f"mlp{l}.{u}"; per[key] = {}
        for n in SIZES:
            s_ = torch.tensor([float(h[tindex[a_]]) for a_, _ in pairs[:n]]); p_ = torch.tensor([float(h[tindex[b_]]) for _, b_ in pairs[:n]])
            ms, mp = float(s_.median()), float(p_.median()); std = float(torch.cat([s_ - s_.mean(), p_ - p_.mean()]).std())
            per[key][str(n)] = {"gap_over_std": (mp - ms) / std, "plural_side_fraction": float((((p_ - ms) * (mp - ms)) > 0).float().mean())}
    report = {"closure_max": closure, "per_unit_by_size": per}
    print(json.dumps(report, indent=1))
    g = lambda k: per[k]["256"]["gap_over_std"]; side = lambda k: per[k]["256"]["plural_side_fraction"]
    report["sharpening_vs_v333"] = {k: abs(g(k)) / abs(V333[k]) for k in per}
    print(json.dumps(report["sharpening_vs_v333"], indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_context_sharpens_3465": abs(g("mlp3.3465")) >= SHARPEN * abs(V333["mlp3.3465"]), "pred_c_context_sharpens_every_surviving_unit": all(abs(g(k)) >= SHARPEN * abs(V333[k]) for k in ("mlp3.493", "mlp5.1036", "mlp8.829")),
                   "pred_d_953_1030_stay_silent": abs(g("mlp8.953")) < SILENT and abs(g("mlp8.1030")) < SILENT, "pred_e_plural_side_fraction_rises": all(side(k) > V333_SIDE[k] for k in V333_SIDE)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "chain_units_in_frame_result_v335", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
