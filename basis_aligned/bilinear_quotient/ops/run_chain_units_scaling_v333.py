#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_closure pred_b_gap_stable_from_16 pred_c_vocab_pairs_replay_lexicon pred_d_plural_side_fraction_high_at_256 pred_e_order_of_units_stable
"""Class-size scaling of the single-token plural separation, 4 -> 256 pairs (v333; Logan: vary the number of tokens per category exponentially). v332: the
six named chain units separate plural from singular nouns from the token alone at 1.5-5.4 pooled std on 64 lexicon pairs. Here the 400 vocabulary s-pairs
of the v287 spec (singular / plural forms found in the vocabulary, not a hand lexicon) at sizes 4 / 16 / 64 / 256, each token alone through blocks 0-8:
per unit and size, the plural - singular median gap in pooled std and the plural-side fraction. Question: does the separation hold for arbitrary
vocabulary pairs and stabilise with size, or was it a lexicon artefact?
PREDICTIONS (scored as written; failures preserved; priors from v332)
    pred_a_closure                        the manual pass reproduces the module hidden within relative 1e-4 (instrument)
    pred_b_gap_stable_from_16             for every unit, |gap(64)| and |gap(256)| are within 30% of each other
    pred_c_vocab_pairs_replay_lexicon     at 64 pairs, every unit's gap has the same sign as in v332 and |gap| >= 0.5x v332's |gap|
    pred_d_plural_side_fraction_high_at_256  at 256 pairs, 3465, 493, 1036 and 829 each put >= 0.85 of plurals on the plural side
    pred_e_order_of_units_stable          the ranking of the six units by |gap| at 256 shares its top-3 with v332's (1036, 829, 3465)
PRICE (registered maximum): 512 single tokens in 2 batches of 256 (blocks 0-8) = 2 forwards; 0 backwards; 0 fits. Bar <= 4.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import dod_battery, dod_units
import run_mlp1_token_table_scaling_v287 as v287

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/chain_units_scaling_v333_result.json"
CANDIDATE_ID = "pronoun_number.chain_units_scaling_v333"
UNITS = {3: (3465, 493), 5: (1036,), 8: (829, 953, 1030)}; TOP = 8
NUMERALS = [" 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9", " two", " three", " four", " five", " six", " seven", " eight", " nine", " 10", " 12", " 20", " 100"]
ONES = [" 1", " one"]
CLOSURE_TOL, STAB, REPLAY_FRAC, SIDE_MIN, SIZES = 1e-4, 0.30, 0.5, 0.85, (4, 16, 64, 256)
V332 = {"mlp3.3465": -4.22, "mlp3.493": 3.8, "mlp5.1036": 5.4, "mlp8.829": 4.35, "mlp8.953": -2.1, "mlp8.1030": 1.48}
V329 = {"singular": -36.755, "plural": -431.818}
FORWARDS_MAX = 4
PREDICTIONS = {"pred_a_closure": "<= 1e-4", "pred_b_gap_stable_from_16": "within 30% x 6", "pred_c_vocab_pairs_replay_lexicon": "same sign, >= 0.5x", "pred_d_plural_side_fraction_high_at_256": ">= 0.85 x 4", "pred_e_order_of_units_stable": "top-3 shared"}


def main() -> None:
    spec = v287.SPEC; pairs = [(a, b) for _, a, b in spec["noun_pairs_vocab"]][:256]
    groups = {"singular": [a for a, _ in pairs], "plural": [b for _, b in pairs]}; tokens = sorted({t for v in groups.values() for t in v})
    plan = {"candidate_id": CANDIDATE_ID, "units": {str(k): list(v) for k, v in UNITS.items()}, "tokens": len(tokens), "groups": {k: len(v) for k, v in groups.items()}, "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only", "bars": {"closure_tol": CLOSURE_TOL, "stab": STAB, "replay_frac": REPLAY_FRAC, "side_min": SIDE_MIN}, "sizes": SIZES}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; forwards = 0
    blocks = model.transformer.h
    H, H2 = {}, {}
    with torch.no_grad():
        for s0 in range(0, len(tokens), 256):
            ids = torch.tensor(tokens[s0:s0 + 256], device="cuda").unsqueeze(1)
            x = F.rms_norm(model.transformer.wte(ids), (model.config.n_embd,)); x0, v1_ = x, None
            for l, block in enumerate(blocks):
                live = block.lambdas[0] * x + block.lambdas[1] * x0; attention, v1_ = block.attn(F.rms_norm(live, (model.config.n_embd,)), v1_); x = live + attention; xin = F.rms_norm(x, (model.config.n_embd,))
                if l in UNITS:
                    hh = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu(); hh2 = dod_units.hidden(model, block.mlp, xin)[:, 0].float().cpu()
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
    g = lambda k, n: per[k][str(n)]["gap_over_std"]
    top3 = set(sorted(per, key=lambda k: -abs(g(k, 256)))[:3])
    predictions = {"pred_a_closure": closure <= CLOSURE_TOL, "pred_b_gap_stable_from_16": all(abs(abs(g(k, 64)) - abs(g(k, 256))) <= STAB * abs(g(k, 256)) for k in per),
                   "pred_c_vocab_pairs_replay_lexicon": all(g(k, 64) * V332[k] > 0 and abs(g(k, 64)) >= REPLAY_FRAC * abs(V332[k]) for k in per), "pred_d_plural_side_fraction_high_at_256": all(per[k]["256"]["plural_side_fraction"] >= SIDE_MIN for k in ("mlp3.3465", "mlp3.493", "mlp5.1036", "mlp8.829")),
                   "pred_e_order_of_units_stable": top3 == {"mlp5.1036", "mlp8.829", "mlp3.3465"}}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "chain_units_scaling_result_v333", "candidate_id": CANDIDATE_ID, "plan": plan, "report": report, "predictions": predictions, "forwards": forwards,
                               "serial_seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
