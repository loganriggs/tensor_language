#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_determiner_sharpens_3465 pred_c_and_suppresses_1036 pred_d_493_insensitive_to_preceding_word pred_e_953_1030_silent_for_every_word
"""Which preceding word sharpens or suppresses the lexical number detectors? (v337). v336: with filler prefixes the detectors' plural / singular separation
on 256 vocabulary pairs followed the word immediately before the noun -- after " and" unit 1036 vanished and 3465 weakened, after " the" / " of" 3465 and 829
peaked. Here one preceding token at a time: "The", " the", " a", ",", " and", " or", " of", " very", " 1", " two", " these", " those" (12 frames), each with the
256 pairs at X; per unit and frame: gap in pooled std and plural-side fraction. Determiners vs coordinators vs plural determiners ("these", "those") vs numerals.
PREDICTIONS (scored as written; failures preserved; priors from v336)
    pred_a_closure                            the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_determiner_sharpens_3465           |gap(3465)| after "The" / " the" / " a" each >= |gap(3465)| alone (2.58)
    pred_c_and_suppresses_1036                |gap(1036)| after " and" and after " or" each <= 0.5 x its alone value (2.01)
    pred_d_493_insensitive_to_preceding_word  493's |gap| stays within 2.0-4.0 std for all 12 frames
    pred_e_953_1030_silent_for_every_word     953 and 1030 stay below 1 pooled std for all 12 frames (including "these" / "those", which are plural themselves)
PRICE (registered maximum): 12 frames x 512 rows / 256 = 24 forwards (blocks 0-8); 0 backwards; 0 fits. Bar <= 26.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/chain_units_preceding_word_v337_result.json"
CANDIDATE_ID = "pronoun_number.chain_units_preceding_word_v337"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, SILENT, ALONE = 1e-4, 1.0, {"mlp3.3465": 2.58, "mlp5.1036": 2.01}
FRAMES = ("The", " the", " a", ",", " and", " or", " of", " very", " 1", " two", " these", " those")
V333 = {"mlp3.3465": -2.58, "mlp3.493": 2.65, "mlp5.1036": 2.01, "mlp8.829": 1.76, "mlp8.953": -0.06, "mlp8.1030": -0.04}
V333_SIDE = {"mlp3.3465": 0.96, "mlp3.493": 0.98, "mlp5.1036": 0.89, "mlp8.829": 0.82}
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 26
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_determiner_sharpens_3465": ">= alone x 3", "pred_c_and_suppresses_1036": "<= 0.5x alone x 2", "pred_d_493_insensitive_to_preceding_word": "2.0-4.0 x 12", "pred_e_953_1030_silent_for_every_word": "< 1 std x 24"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "silent": SILENT, "alone": ALONE}, "frames": list(FRAMES)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    blocks = model.transformer.h
    per = {}; closure = 0.0
    for frame in FRAMES:
        k = 1; pre = [L._single(frame)]
        H, H2 = {}, {}
        with torch.no_grad():
            for s0 in range(0, len(tokens), 256):
                ids = torch.tensor([pre + [t] for t in tokens[s0:s0 + 256]], device="cuda")
                x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
                for l, block in enumerate(blocks):
                    live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                    if l in UNITS:
                        hh = dod_units.hidden(model, block.mlp, xin)[:, k].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, k].float().cpu()
                        for u in UNITS[l]: H.setdefault((l, u), []).append(hh[:, u]); H2.setdefault((l, u), []).append(hh2[:, u])
                    if l == TOP: break
                    x = x + block.mlp(xin)
                forwards += 1
        H = {kk: torch.cat(v) for kk, v in H.items()}; H2 = {kk: torch.cat(v) for kk, v in H2.items()}
        closure = max(closure, max(float(((H2[kk] - H[kk]).abs() / H[kk].abs().clamp_min(1e-6)).max()) for kk in H)); tindex = {t: i for i, t in enumerate(tokens)}
        for (l, u), h in H.items():
            s_ = torch.tensor([float(h[tindex[a_]]) for a_, _ in pairs]); p_ = torch.tensor([float(h[tindex[b_]]) for _, b_ in pairs])
            ms, mp = float(s_.median()), float(p_.median()); std = float(torch.cat([s_ - s_.mean(), p_ - p_.mean()]).std())
            per.setdefault(f"mlp{l}.{u}", {})[frame] = {"gap_over_std": (mp - ms) / std, "plural_side_fraction": float((((p_ - ms) * (mp - ms)) > 0).float().mean())}
    report = {"closure_max": closure, "per_unit_by_frame": per}
    print(json.dumps(report, indent=1))
    g = lambda u, f: abs(per[u][f]["gap_over_std"])
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_determiner_sharpens_3465": all(g("mlp3.3465", f) >= ALONE["mlp3.3465"] for f in ("The", " the", " a")), "pred_c_and_suppresses_1036": all(g("mlp5.1036", f) <= 0.5 * ALONE["mlp5.1036"] for f in (" and", " or")),
                   "pred_d_493_insensitive_to_preceding_word": all(2.0 <= g("mlp3.493", f) <= 4.0 for f in FRAMES), "pred_e_953_1030_silent_for_every_word": all(g(u, f) < SILENT for u in ("mlp8.953", "mlp8.1030") for f in FRAMES)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "chain_units_preceding_word_result_v337", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
