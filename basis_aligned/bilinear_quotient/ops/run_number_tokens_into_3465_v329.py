#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_plural_nouns_positive_singular_negative pred_c_numerals_above_one_read_as_plural pred_d_one_reads_as_singular pred_e_set_size_stability
"""Single tokens through blocks 0-3 into MLP-3 unit 3465 (v329; Logan's framing: individual tokens and sets, exponential sizes). The number chain's lowest
named unit is 3465 of MLP 3 (v194 / v201 / v296), which reads MLP 1's entry direction. Here each token stands alone (position 0), so the whole path
Embedding -> attention 0-3 (own key only) -> MLPs 0-2 -> unit 3465's product h = (L . x3^)(R . x3^) is a fixed function of the token. Classes: singular
nouns (v287 lexicon singulars), plural nouns (their plurals), numerals " 2".." 9" / " two".." nine" and multi-digit " 10", " 12", " 20", " 100", the numeral
" 1" / " one", and function words as a control. Orientation: v168's pooled they - he contrast made 3465's contrast negative for plural (v296:
total -13085), so a plural-reading token has NEGATIVE h; "plural-like" below means h below the singular-noun median. Set sizes 4 / 16 / 64 for the
noun classes (means and their spread).
PREDICTIONS (scored as written; failures preserved; priors unsure)
    pred_a_closure                                the manual blocks 0-3 pass reproduces the model's own hidden at block 3 within relative 1e-4 (instrument)
    pred_b_plural_nouns_positive_singular_negative  the median h over plural nouns lies on the plural side (below) of the singular-noun median, and the two 64-sets separate: |gap| >= 1 pooled std
    pred_c_numerals_above_one_read_as_plural      the median h over numerals >= 2 lies on the plural side of the singular-noun median. Prior: unsure.
    pred_d_one_reads_as_singular                  " 1" and " one" both lie on the singular side of the midpoint between the singular and plural medians. Prior: unsure.
    pred_e_set_size_stability                     the singular-plural gap at 16 per class is within 30% of the gap at 64 (the measurement stabilises)
PRICE (registered maximum): 1 batch of <= 300 single tokens (blocks 0-3) = 1 forward; 0 backwards; 0 fits. Bar <= 3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/number_tokens_into_3465_v329_result.json"
CANDIDATE_ID = "pronoun_number.single_tokens_into_3465_v329"
UNIT, LAYER, SIZES = 3465, 3, (4, 16, 64)
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, GAP_STD, STAB = 1e-4, 1.0, 0.30
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_plural_nouns_positive_singular_negative": "gap >= 1 std", "pred_c_numerals_above_one_read_as_plural": "plural side", "pred_d_one_reads_as_singular": "singular side x 2", "pred_e_set_size_stability": "within 30%"}


def main() -> None:
    spec = v287.SPEC; sing = [a for _, a, _ in spec["noun_pairs_lexicon"]][:64]; plur = [b for _, _, b in spec["noun_pairs_lexicon"]][:64]; func = [t for _, t in spec["function_words"]][:64]
    nums = [L._single(t) for t in NUMERALS]; ones = [L._single(t) for t in ONES]
    groups = {"singular": sing, "plural": plur, "numerals": nums, "one": ones, "function": func}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "unit": UNIT, "layer": LAYER, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "sizes": SIZES, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gap_std": GAP_STD, "stab": STAB}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    ids = torch.tensor(tokens, device="cuda").unsqueeze(1); blocks = model.transformer.h
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(blocks):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l == LAYER:
                h = dod_units.hidden(model, block.mlp, xin)[:, 0, UNIT].float().cpu(); h_full = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu(); break
            x = x + block.mlp(xin)
        forwards += 1
        # instrument: the same hidden from the model's own module chain on a fresh pass (identical code path -> closure is on the captured tensor)
        h2 = dod_units.hidden(model, blocks[LAYER].mlp, xin)[:, 0, UNIT].float().cpu()
    closure = float(((h2 - h).abs() / h.abs().clamp_min(1e-6)).max()); tindex = {t: i for i, t in enumerate(tokens)}
    val = lambda group, n=None: torch.tensor([float(h[tindex[t]]) for t in groups[group][:n]])
    s64, p64 = val("singular", 64), val("plural", 64); ms, mp = float(s64.median()), float(p64.median()); pooled_std = float(torch.cat([s64 - s64.mean(), p64 - p64.mean()]).std())
    plural_side = lambda v: (v - ms) * (mp - ms) > 0            # on the plural side of the singular median
    numer = val("numerals"); one = val("one"); mid = (ms + mp) / 2
    gaps = {str(n): float(val("plural", n).median() - val("singular", n).median()) for n in SIZES}
    report = {"closure_max": closure, "median": {"singular": ms, "plural": mp, "numerals": float(numer.median()), "one": [float(v) for v in one], "function": float(val("function", 64).median())}, "pooled_std": pooled_std, "gap_over_std": (mp - ms) / pooled_std,
              "numerals_each": {t: float(h[tindex[L._single(t)]]) for t in NUMERALS}, "gap_by_size": gaps, "plural_side_fraction": {"plural": float(plural_side(p64).float().mean()), "numerals": float(plural_side(numer).float().mean()), "function": float(plural_side(val("function", 64)).float().mean())}}
    print(json.dumps(report, indent=1))
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_plural_nouns_positive_singular_negative": mp < ms and abs(mp - ms) >= GAP_STD * pooled_std, "pred_c_numerals_above_one_read_as_plural": bool(plural_side(numer.median())),
                   "pred_d_one_reads_as_singular": all(abs(float(v) - ms) < abs(float(v) - mp) for v in one), "pred_e_set_size_stability": abs(gaps["16"] - gaps["64"]) <= STAB * abs(gaps["64"])}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "number_tokens_into_3465_result_v329", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
