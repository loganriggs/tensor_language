#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_baseline_replays pred_b_noun_alone_carries_most pred_c_other_slots_small pred_d_slots_sub_additive pred_e_mean_ablation_near_zero_margin
"""Fixed-position forward folding: the slot's role (v604; approved 19 Sep 22:12 UTC). Logan's phrasing: "fix all positions but one and fold forward
(the slot's role)". Method: for a natural sentence, replace EVERY position's embedding except one with the position-wise MEAN embedding taken over the
122 natural number rows (a content-free "average token" reference, not zero -- zero is off-distribution for an RMSNorm'd model since it is a fixed
point of the norm only in the limit), leaving position t's own embedding untouched, run the real forward, and read the they-he margin. Doing this once
per position (T ~ 24 positions) gives a SLOT IMPORTANCE PROFILE: how much of the true margin survives when only that one slot is allowed to speak. Also
run the reference "all positions mean-ablated" (should be near zero) and "no positions ablated" (native, replays) as endpoints of the profile.
PREDICTIONS (scored as written; failures preserved; priors from the number circuit's known noun-centrality)
    pred_a_baseline_replays          the unedited manual forward replays the model's they - he margin within 1e-3, for every sentence
    pred_b_noun_alone_carries_most   the noun-position-alone margin is the single largest slot value, for every sentence (the number circuit is known
                                      to run through the noun)
    pred_c_other_slots_small         every non-noun slot alone closes <= 0.30 of the native margin, for every sentence. Prior: unsure
    pred_d_slots_sub_additive        summing all single-slot margins is LESS than the native margin, for every sentence (slots are not simply additive
                                      contributions -- the model needs several positions present together, matching the mixed-partials finding)
    pred_e_mean_ablation_near_zero_margin  with every position mean-ablated the margin is within 0.15 x the native margin's magnitude, for every
                                      sentence (the content-free reference carries little number signal on its own)
PRICE (registered maximum): 4 sentences x (~24 slot forwards + 2 endpoint forwards + 1 native + 1 replay-check forward) ~= 112 forwards; 0 backwards; 0 fits. Bar <= 120.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, time
import aspectual_dod_lib as L
import circuit_fast_screen_producer as producer
import run_value_copy_writers_v406 as v406
import run_pronoun_number_dod_battery_v76 as g
import dod_battery

ROOT = dod_battery.ROOT
OUT = ROOT / "circuits/followups/fixed_position_slot_role_v604_result.json"
CANDIDATE_ID = "slot_role.fixed_position_v604"
N_SENT = 4
REPLAY_TOL, OTHER_MAX, MEAN_MAX = 1e-3, 0.30, 0.15
FORWARDS_MAX = 120
PREDICTIONS = {"pred_a_baseline_replays": "<= 1e-3", "pred_b_noun_alone_carries_most": "noun slot is max x N", "pred_c_other_slots_small": "<= 0.30 x N",
               "pred_d_slots_sub_additive": "sum < native x N", "pred_e_mean_ablation_near_zero_margin": "<= 0.15 x native x N"}


def main() -> None:
    recs = [r_ for p_ in v406.NATURAL for r_ in json.loads(p_.read_text())["rows"]][:N_SENT]
    plan = {"candidate_id": CANDIDATE_ID, "sentences": len(recs), "forwards_max": FORWARDS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only",
            "bars": {"replay_tol": REPLAY_TOL, "other_max": OTHER_MAX, "mean_max": MEAN_MAX}}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    t0 = time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda"); torch, F, model = backend.torch, backend.F, backend.model; blocks = model.transformer.h
    D = model.config.n_embd; THEY, HE = L._single(" they"), L._single(" he")
    forwards = 0

    def margin_forward(e):   # e: [1, T, D] fixed embedding tensor, no grad needed
        with torch.no_grad():
            x = F.rms_norm(e, (D,)); x0, v1_ = x, None
            for block in blocks:
                live = block.lambdas[0] * x + block.lambdas[1] * x0
                attention, v1_ = block.attn(F.rms_norm(live, (D,)), v1_)
                x = live + attention
                x = x + block.mlp(F.rms_norm(x, (D,)))
            final = F.rms_norm(x, (D,))
            z = 30.0 * torch.tanh(model.lm_head(final) / 30.0)
            return float(z[0, -1, THEY] - z[0, -1, HE])

    # a position-wise mean embedding, averaged over the 122 natural number rows' own tokens at each of the first 24 positions (content-free reference)
    rows, he, she, agents, objects = g.build()
    all_ids = [r_["ids"] for r_ in recs] + [list(row.ids) for row in rows]
    max_len = max(len(s) for s in all_ids)
    with torch.no_grad():
        acc = torch.zeros(max_len, D, device="cuda"); counts = torch.zeros(max_len, device="cuda")
        for s in all_ids:
            t = torch.tensor(s, device="cuda"); e = model.transformer.wte(t).float()
            acc[:len(s)] += e; counts[:len(s)] += 1
        mean_embed = acc / counts.clamp_min(1).unsqueeze(-1)   # [max_len, D]

    reports = []
    for r_ in recs:
        ids = r_["ids"]; T = len(ids); noun_pos = r_["cue_offset"]
        tokens = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            e_native = model.transformer.wte(tokens).float()
        native_margin = margin_forward(e_native); forwards += 1
        hook = {}; hh = model.lm_head.register_forward_hook(lambda m, a, o: hook.setdefault("z", o)); model(tokens, tokens.clone().contiguous()); hh.remove(); forwards += 1
        true_z = hook["z"].float()[0, -1]; true_z = 30 * torch.tanh(true_z / 30); true_margin = float(true_z[THEY] - true_z[HE])
        replay_err = abs(native_margin - true_margin)
        e_ref = mean_embed[:T].unsqueeze(0).clone()
        all_ablated_margin = margin_forward(e_ref); forwards += 1
        slots = {}
        for t in range(T):
            e_t = e_ref.clone(); e_t[0, t] = e_native[0, t]
            slots[t] = margin_forward(e_t); forwards += 1
        slot_sum = sum(slots.values())
        noun_is_max = max(slots, key=lambda k: abs(slots[k])) == noun_pos
        other_max = max(abs(v / native_margin) for k, v in slots.items() if k != noun_pos) if abs(native_margin) > 1e-6 else 0.0
        reports.append({"noun_pos": noun_pos, "T": T, "native_margin": native_margin, "true_margin": true_margin, "replay_err": replay_err, "all_ablated_margin": all_ablated_margin, "slots": slots,
                         "slot_sum": slot_sum, "noun_is_max_slot": noun_is_max, "other_slots_max_fraction": other_max,
                         "noun_slot_fraction": slots[noun_pos] / native_margin if abs(native_margin) > 1e-6 else None})
        print(f"native={native_margin:.3f} ablated={all_ablated_margin:.3f} noun_slot={slots[noun_pos]:.3f} noun_is_max={noun_is_max} slot_sum={slot_sum:.3f} other_max_frac={other_max:.3f}")

    predictions = {"pred_a_baseline_replays": all(r["replay_err"] <= REPLAY_TOL for r in reports),
                   "pred_b_noun_alone_carries_most": all(r["noun_is_max_slot"] for r in reports),
                   "pred_c_other_slots_small": all(r["other_slots_max_fraction"] <= OTHER_MAX for r in reports),
                   "pred_d_slots_sub_additive": all(abs(r["slot_sum"]) < abs(r["native_margin"]) for r in reports),
                   "pred_e_mean_ablation_near_zero_margin": all(abs(r["all_ablated_margin"]) <= MEAN_MAX * abs(r["native_margin"]) for r in reports)}
    if forwards > FORWARDS_MAX:
        raise SystemExit(f"price exceeded: {forwards} > {FORWARDS_MAX}")
    OUT.write_text(json.dumps({"schema": "fixed_position_slot_role_result_v604", "candidate_id": CANDIDATE_ID, "plan": plan, "reports": reports,
                               "predictions": predictions, "forwards": forwards, "serial_seconds": time.perf_counter() - t0,
                               "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "forwards": forwards}, indent=2))


if __name__ == "__main__":
    main()
