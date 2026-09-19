#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_gap_peaks_at_short_context pred_c_gap_survives_8_tokens pred_d_plural_side_fraction_stays_high pred_e_953_1030_silent_at_every_length
"""Context-length scaling of the lexical detectors (v336; Logan: vary exponentially). v333 / v335: the 256 vocabulary s-pairs separate on 3465 / 493 / 1036 / 829
at 2-3 std from the token alone and 2.4-3.8 std after one context token ("The"). Here k = 0, 1, 2, 4, 8 filler tokens (", and of the very" cyclic; k = 1
is "," not "The") precede X; per unit and k: gap in pooled std and plural-side fraction at 256 pairs. Question: does the sharpening continue, saturate or
reverse as context grows -- MLP 1's lookup gain falls with k (v291), so the detectors' input identity content falls too.
PREDICTIONS (scored as written; failures preserved; priors from v291 / v335)
    pred_a_closure                        the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_gap_peaks_at_short_context     for 3465, |gap| is largest at k = 1 or 2 (one or two tokens of context sharpen; more dilutes). Prior: unsure.
    pred_c_gap_survives_8_tokens          at k = 8 every one of 3465 / 493 / 1036 / 829 still separates by >= 1.5 pooled std
    pred_d_plural_side_fraction_stays_high  at k = 8 each of the four keeps >= 0.90 of plurals on the plural side
    pred_e_953_1030_silent_at_every_length  953 and 1030 stay below 1 pooled std at every k
PRICE (registered maximum): 5 lengths x 512 rows / 256 = 10 forwards (blocks 0-8); 0 backwards; 0 fits. Bar <= 12.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/chain_units_context_length_v336_result.json"
CANDIDATE_ID = "pronoun_number.chain_units_context_length_v336"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, SILENT, SURVIVE, SIDE_MIN, SIZES, LENGTHS = 1e-4, 1.0, 1.5, 0.90, (256,), (0, 1, 2, 4, 8)
PHRASE = (",", " and", " of", " the", " very")
V333 = {"mlp3.3465": -2.58, "mlp3.493": 2.65, "mlp5.1036": 2.01, "mlp8.829": 1.76, "mlp8.953": -0.06, "mlp8.1030": -0.04}
V333_SIDE = {"mlp3.3465": 0.96, "mlp3.493": 0.98, "mlp5.1036": 0.89, "mlp8.829": 0.82}
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 12
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_gap_peaks_at_short_context": "k = 1 or 2", "pred_c_gap_survives_8_tokens": ">= 1.5 std x 4", "pred_d_plural_side_fraction_stays_high": ">= 0.90 x 4", "pred_e_953_1030_silent_at_every_length": "< 1 std x 2 x 5"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "silent": SILENT, "survive": SURVIVE, "side_min": SIDE_MIN}, "lengths": LENGTHS, "phrase": list(PHRASE)}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    blocks = model.transformer.h
    fill = [L._single(t) for t in PHRASE]; filler = lambda k: [fill[i % len(fill)] for i in range(k)]; per = {}; closure = 0.0
    for k in LENGTHS:
        H, H2 = {}, {}
        with torch.no_grad():
            for s0 in range(0, len(tokens), 256):
                ids = torch.tensor([filler(k) + [t] for t in tokens[s0:s0 + 256]], device="cuda")
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
            per.setdefault(f"mlp{l}.{u}", {})[str(k)] = {"gap_over_std": (mp - ms) / std, "plural_side_fraction": float((((p_ - ms) * (mp - ms)) > 0).float().mean())}
    report = {"closure_max": closure, "per_unit_by_length": per}
    print(json.dumps(report, indent=1))
    g = lambda u, k: per[u][str(k)]["gap_over_std"]; side = lambda u, k: per[u][str(k)]["plural_side_fraction"]; four = ("mlp3.3465", "mlp3.493", "mlp5.1036", "mlp8.829")
    peak = max(LENGTHS, key=lambda k: abs(g("mlp3.3465", k)))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_gap_peaks_at_short_context": peak in (1, 2), "pred_c_gap_survives_8_tokens": all(abs(g(u, 8)) >= SURVIVE for u in four),
                   "pred_d_plural_side_fraction_stays_high": all(side(u, 8) >= SIDE_MIN for u in four), "pred_e_953_1030_silent_at_every_length": all(abs(g(u, k)) < SILENT for u in ("mlp8.953", "mlp8.1030") for k in LENGTHS)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "chain_units_context_length_result_v336", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
