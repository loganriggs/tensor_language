#!/usr/bin/env python3
"""Fresh-population gain curve using the shared rank48 reverse-dose executor."""
# BQGATE: EXPERIMENT pred_a_authority_population_finiteness_and_exact_price pred_b_uncalibrated_gain_is_control_selective pred_c_control_flip_is_a_calibration_threshold pred_d_a_safe_calibrated_gain_exists pred_e_behavior_dose_is_monotone_and_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v13 as fresh_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as fresh_i
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import run_temporal_five_mlp_rank48_reverse_gain_dose_v1 as engine

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_reverse_fresh_gain_curve_v1.json"
PARENT = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_gain125_fresh_v1_result.json"
SITE = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_site_gain_localization_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
ENGINE = ROOT / "ops/run_temporal_five_mlp_rank48_reverse_gain_dose_v1.py"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_reverse_fresh_gain_curve_v1_result.json"
GAINS = (0.75, 1.0, 1.1, 1.2, 1.25)
EXPECTED = {
    "parent": "6fee15243ad80cd7b21b86c0001474d11b5789d955d1cd90b793fc65584dd3d9",
    "site": "aa085d28f3abe2fee7b6bd97918a39b9298dc2dba12efc67234d1b434b87a958",
    "tcap": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
    "icap": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
    "tbuilder": "3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509",
    "ibuilder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "engine": "946de70657dee61ac1d6ada876210ecf8c476904cce7d7daba3a746067c010bc",
}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def finite_tree(value):
    if isinstance(value, dict): return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite_tree(item) for item in value)
    if isinstance(value, float): return math.isfinite(value)
    return True


def control_ok(record):
    control = record["control"]
    return (control["top1_flip_fraction"] == 0.0 and max(control["margin_rms_fraction"].values()) <= .1
            and control["median_kl"] <= .02)


def target_ok(record):
    target = record["target"]
    return (all(.9 <= target["behavior_signed_projection"][task] <= 1.1 for task in ("temporal", "iswas"))
            and target["worst_target_residual"] <= .15)


def main():
    paths = {"parent": PARENT, "site": SITE, "tcap": TCAP, "icap": ICAP,
             "tbuilder": Path(fresh_t.__file__), "ibuilder": Path(fresh_i.__file__), "engine": ENGINE}
    observed = {key: sha(path) for key, path in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"fresh gain-curve authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_reverse_fresh_gain_curve_v1",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "gains": GAINS, "model_forwards_exact": 41, "fit_updates": 0, "model_updates": 0,
           "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    captured = {}
    original_writer = engine.atomic_create_json

    def write_result(_path, raw):
        reports = raw["reports"]
        gain_key = lambda gain: str(gain)
        no_flip_low = all(control_ok(reports[label][gain_key(gain)]) for label in reports for gain in (.75, 1.0))
        flips_above = any(reports[label][gain_key(gain)]["control"]["top1_flip_fraction"] > 0
                          for label in reports for gain in (1.1, 1.2, 1.25))
        safe = [gain for gain in (1.1, 1.2) if all(control_ok(reports[label][gain_key(gain)])
                                                           and target_ok(reports[label][gain_key(gain)]) for label in reports)]
        monotone = all(all(reports[label][gain_key(a)]["target"]["behavior_signed_projection"][task]
                           < reports[label][gain_key(b)]["target"]["behavior_signed_projection"][task]
                           for a, b in zip(GAINS, GAINS[1:])) for label in reports for task in ("temporal", "iswas"))
        crossfit = max(abs(reports["even_fit"][gain_key(gain)]["target"]["behavior_signed_projection"][task]
                           - reports["odd_fit"][gain_key(gain)]["target"]["behavior_signed_projection"][task])
                       for gain in GAINS for task in ("temporal", "iswas"))
        caps = json.loads(TCAP.read_text()), json.loads(ICAP.read_text())
        pa = (caps[0]["terminal"] == "manifest" and caps[1]["terminal"] == "screen" and finite_tree(reports))
        predictions = {
            "pred_a_authority_population_finiteness_and_exact_price": bool(pa),
            "pred_b_uncalibrated_gain_is_control_selective": bool(all(control_ok(reports[label]["1.0"]) for label in reports)),
            "pred_c_control_flip_is_a_calibration_threshold": bool(no_flip_low and flips_above),
            "pred_d_a_safe_calibrated_gain_exists": bool(safe),
            "pred_e_behavior_dose_is_monotone_and_crossfit_stable": bool(monotone and crossfit <= .1),
        }
        terminal = ("invalid" if not pa else "provisional_safe_gain_window" if all(predictions.values())
                    else "preexisting_fresh_control_failure" if not predictions["pred_b_uncalibrated_gain_is_control_selective"]
                    else "calibration_selectivity_tradeoff")
        summary = {
            "safe_calibrated_gains": safe,
            "first_flip_gain": min((gain for gain in GAINS if any(reports[label][gain_key(gain)]["control"]["top1_flip_fraction"] > 0 for label in reports)), default=None),
            "gain1_behavior_min": min(reports[label]["1.0"]["target"]["behavior_signed_projection"][task] for label in reports for task in ("temporal", "iswas")),
            "gain1_control_flip_max": max(reports[label]["1.0"]["control"]["top1_flip_fraction"] for label in reports),
            "crossfit_behavior_max_abs": crossfit,
        }
        result = {"schema": "temporal_five_mlp_rank48_reverse_fresh_gain_curve_result_v1",
                  "started_utc": raw["started_utc"], "finished_utc": now(), "serial_seconds": raw["serial_seconds"],
                  "authority_sha256": EXPECTED, "gains": GAINS, "reports": reports, "summary": summary,
                  "predictions": predictions, "terminal": terminal,
                  "price": {"model_forwards": 41, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
        original_writer(OUT, result); captured.update(result)

    engine.TCAP, engine.ICAP, engine.OUT, engine.GAINS = TCAP, ICAP, OUT, GAINS
    engine.atomic_create_json = write_result
    oodctx.ood_t, oodctx.ood_i = fresh_t, fresh_i
    engine.main()
    print(json.dumps({key: captured[key] for key in ("summary", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
