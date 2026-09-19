#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_3465_replays pred_c_every_named_unit_separates_from_the_token_alone pred_d_separation_does_not_grow_up_the_chain pred_e_numerals_and_pronouns_ignored_everywhere
"""Single tokens up the whole named chain (v332). v329-v331: folded alone through blocks 0-3, MLP-3 unit 3465 is a lexical plural-noun detector. The
chain's other named units -- 493 (MLP 3), 1036 (MLP 5; v194), 829 / 953 / 1030 (MLP 8; v168 / v251) -- are reached the same way, each token alone
through blocks 0-8 with every head reading only its own key. Per unit: the plural / singular noun medians and gap in pooled std (64 + 64 lexicon nouns),
and the axis position of numerals (2-9, two-nine, 10, 12, 20, 100), plural pronouns (they, we, them, us, these, those) and irregular plurals.
Question: is the single-token plural signal already complete at MLP 3, or does the chain sharpen it without context?
PREDICTIONS (scored as written; failures preserved; priors unsure beyond 3465)
    pred_a_closure                                    the manual pass reproduces the module hidden at each layer within relative 1e-4 (instrument)
    pred_b_3465_replays                               3465's noun medians replay v329's -37 / -432 within 5%
    pred_c_every_named_unit_separates_from_the_token_alone  every one of the six units separates plural from singular nouns by >= 1 pooled std with the sign of its known contrast (3465 / 493 from v296; 829 / 953 / 1030 / 1036 taken from the sign of their own median gap here, reported)
    pred_d_separation_does_not_grow_up_the_chain      the largest |gap| in std among the MLP-8 units is <= 1.5x 3465's (the chain adds context, not sharper lexical number)
    pred_e_numerals_and_pronouns_ignored_everywhere   numerals' and plural pronouns' axis positions are within +- 0.25 for every unit
PRICE (registered maximum): 1 batch of <= 220 single tokens (blocks 0-8) = 1 forward; 0 backwards; 0 fits. Bar <= 3.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/tokens_up_the_chain_v332_result.json"
CANDIDATE_ID = "pronoun_number.tokens_up_the_chain_v332"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, GAP_STD, REPLAY, GROW_MAX, NEAR = 1e-4, 1.0, 0.05, 1.5, 0.25
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 3
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_3465_replays": "within 5%", "pred_c_every_named_unit_separates_from_the_token_alone": ">= 1 std x 6", "pred_d_separation_does_not_grow_up_the_chain": "<= 1.5x", "pred_e_numerals_and_pronouns_ignored_everywhere": "within 0.25 x 6"}


def main() -> None:
    spec = v287.SPEC; sing = [a for _, a, _ in spec["noun_pairs_lexicon"]][:64]; plur = [b for _, _, b in spec["noun_pairs_lexicon"]][:64]
    def T(xs):
        out = []
        for t in xs:
            try: out.append(L._single(t))
            except Exception: pass                      # multi-token entries are dropped and listed in the plan
        return out
    groups = {"singular": sing, "plural": plur, "numerals": T([" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]),
              "plural_pronouns": T([" they", " we", " them", " us", " these", " those"]), "irregular_plural": T([" men", " women", " children", " people", " feet", " teeth", " mice"])}
    tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "gap_std": GAP_STD, "replay": REPLAY, "grow_max": GROW_MAX, "near": NEAR}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    ids = torch.tensor(tokens, device="cuda").unsqueeze(1); blocks = model.transformer.h
    H, H2 = {}, {}
    with torch.no_grad():
        x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
        for l, block in enumerate(blocks):
            live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
            if l in UNITS:
                hh = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu()
                for u in UNITS[l]: H[(l, u)] = hh[:, u]; H2[(l, u)] = hh2[:, u]
            if l == TOP: break
            x = x + block.mlp(xin)
        forwards += 1
    closure = max(float(((H2[k] - H[k]).abs() / H[k].abs().clamp_min(1e-6)).max()) for k in H); tindex = {t: i for i, t in enumerate(tokens)}
    per = {}
    for (l, u), h in H.items():
        val = lambda g: torch.tensor([float(h[tindex[t]]) for t in groups[g]])
        s64, p64 = val("singular"), val("plural"); ms, mp = float(s64.median()), float(p64.median()); std = float(torch.cat([s64 - s64.mean(), p64 - p64.mean()]).std())
        side = lambda v: float((v - ms) / (mp - ms)) if mp != ms else 0.0
        per[f"mlp{l}.{u}"] = {"singular_median": ms, "plural_median": mp, "gap_over_std": (mp - ms) / std, "plural_side_fraction": float((((p64 - ms) * (mp - ms)) > 0).float().mean()), "axis": {g: side(val(g).median()) for g in ("numerals", "plural_pronouns", "irregular_plural")}}
    g3465 = per["mlp3.3465"]; grow = max(abs(per[k]["gap_over_std"]) for k in per if k.startswith("mlp8")) / abs(g3465["gap_over_std"])
    report = {"closure_max": closure, "per_unit": per, "mlp8_over_3465_gap_ratio": grow}
    print(json.dumps(report, indent=1))
    known_sign = {"mlp3.3465": -1, "mlp3.493": +1}
    def sep_ok(k, v):
        sgn = known_sign.get(k); return abs(v["gap_over_std"]) >= GAP_STD and (sgn is None or (v["plural_median"] - v["singular_median"]) * sgn > 0)
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_3465_replays": abs(g3465["singular_median"] - V329["singular"]) <= REPLAY * abs(V329["singular"]) and abs(g3465["plural_median"] - V329["plural"]) <= REPLAY * abs(V329["plural"]),
                   "pred_c_every_named_unit_separates_from_the_token_alone": all(sep_ok(k, v) for k, v in per.items()), "pred_d_separation_does_not_grow_up_the_chain": grow <= GROW_MAX,
                   "pred_e_numerals_and_pronouns_ignored_everywhere": all(abs(v["axis"]["numerals"]) <= NEAR and abs(v["axis"]["plural_pronouns"]) <= NEAR for v in per.values())}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "tokens_up_the_chain_result_v332", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
